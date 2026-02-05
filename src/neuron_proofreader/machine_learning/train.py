"""
Created on Wed July 25 16:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for a custom class for training neural networks to perform neuron
proofreading classification tasks.

"""

from datetime import datetime
from sklearn.metrics import precision_score, recall_score, accuracy_score
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import numpy as np
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from neuron_proofreader.utils import img_util, ml_util, util


class Trainer:
    """
    Trainer class for training a model to perform binary classifcation.

    Attributes
    ----------
    best_f1 : float
        Best F1 score achieved so far on valiation dataset.
    criterion : torch.nn.BCEWithLogitsLoss
        Loss function used during training.
    device : str, optional
        Device that model is run on.
    log_dir : str
        Path to directory that tensorboard and checkpoints are saved to.
    max_epochs : int
        Maximum number of training epochs.
    min_recall : float, optional
        Minimum recall required for model checkpoints to be saved.
    model : torch.nn.Module
        Model that is trained to perform binary classification.
    model_name : str
        Name of model used for logging and checkpointing.
    optimizer : torch.optim.AdamW
        Optimizer that is used during training.
    save_mistake_mips : bool, optional
        Indication of whether to save MIPs of mistakes.
    scheduler : torch.optim.lr_scheduler.CosineAnnealingLR
        Scheduler used to the adjust learning rate.
    writer : torch.utils.tensorboard.SummaryWriter
        Writer object that writes to a tensorboard.
    """

    def __init__(
        self,
        model,
        model_name,
        output_dir,
        device="cuda",
        lr=1e-3,
        max_epochs=200,
        min_recall=0,
        save_mistake_mips=False
    ):
        """
        Instantiates a Trainer object.

        Parameters
        ----------
        model : torch.nn.Module
            Model that is trained to perform binary classification.
        model_name : str
            Name of model used for logging and checkpointing.
        output_dir : str
            Directory that tensorboard and model checkpoints are written to.
        lr : float, optional
            Learning rate. Default is 1e-3.
        max_epochs : int, optional
            Maximum number of training epochs. Default is 200.
        min_recall : float, optional
            Minimum recall required for model checkpoints to be saved. Default
            is 0.
        save_mistake_mips : bool, optional
            Indication of whether to save MIPs of mistakes. Default is False.
        """
        # Set experiment name
        exp_name = "session-" + datetime.today().strftime("%Y%m%d_%H%M")
        log_dir = os.path.join(output_dir, exp_name)
        util.mkdir(log_dir)

        # Instance attributes
        self.best_f1 = 0
        self.device = device
        self.log_dir = log_dir
        self.max_epochs = max_epochs
        self.min_recall = min_recall
        self.mistakes_dir = os.path.join(log_dir, "mistakes")
        self.model_name = model_name
        self.save_mistake_mips = save_mistake_mips

        self.criterion = nn.BCEWithLogitsLoss()
        self.model = model.to(device)
        # Only pass trainable parameters to optimizer (handles frozen encoder)
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max_epochs)
        self.writer = SummaryWriter(log_dir=log_dir)

    # --- Core Routines ---
    def run(self, train_dataloader, val_dataloader):
        """
        Runs the full training and validation loop.

        Parameters
        ----------
        train_dataloader : torch.utils.data.Dataset
            Dataloader used for training.
        val_dataloader : torch.utils.data.Dataset
            Dataloader used for validation.
        """
        exp_name = os.path.basename(os.path.normpath(self.log_dir))
        print("\nExperiment:", exp_name)
        for epoch in range(self.max_epochs):
            # Train-Validate
            train_stats = self.train_step(train_dataloader, epoch)
            val_stats = self.validate_step(val_dataloader, epoch)
            new_best = self.check_model_performance(val_stats, epoch)

            # Report reuslts
            print(f"\nEpoch {epoch}: " + ("New Best!" if new_best else " "))
            self.report_stats(train_stats, is_train=True)
            self.report_stats(val_stats, is_train=False)

            # Step scheduler
            self.scheduler.step()

    def train_step(self, train_dataloader, epoch):
        """
        Performs a single training epoch over the provided DataLoader.

        Parameters
        ----------
        train_dataloader : torch.utils.data.DataLoader
            DataLoader for the training dataset.
        epoch : int
            Current training epoch.

        Returns
        -------
        stats : Dict[str, float]
            Dictionary of aggregated training metrics.
        """
        self.model.train()
        loss, y, hat_y = list(), list(), list()
        rank = getattr(self, "rank", 0)
        pbar = tqdm(train_dataloader, desc="Training", unit="batch", disable=rank != 0)
        for x_i, y_i in pbar:
            # Forward pass
            self.optimizer.zero_grad()
            hat_y_i, loss_i = self.forward_pass(x_i, y_i)

            # Backward pass
            self.scaler.scale(loss_i).backward()

            # Step optimizer
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Store results
            y.extend(ml_util.to_cpu(y_i, True).flatten().tolist())
            hat_y.extend(ml_util.to_cpu(hat_y_i, True).flatten().tolist())
            loss.append(float(ml_util.to_cpu(loss_i)))
            if rank == 0:
                pbar.set_postfix(loss=f"{loss[-1]:.4f}")

        # Write stats to tensorboard
        stats = self.compute_stats(y, hat_y)
        stats["loss"] = np.mean(loss)
        self.update_tensorboard(stats, epoch, "train_")
        return stats

    def validate_step(self, val_dataloader, epoch):
        """
        Performs a full validation loop over the given dataloader.

        Parameters
        ----------
        val_dataloader : torch.utils.data.DataLoader
            DataLoader for the validation dataset.
        epoch : int
            Current training epoch.

        Returns
        -------
        stats : Dict[str, float]
            Dictionary of aggregated validation metrics.
        is_best : bool
            True if the current F1 score is the best so far.
        """
        # Initializations
        idx_offset = 0
        loss_accum = 0
        y_accum = list()
        hat_y_accum = list()
        rank = getattr(self, "rank", 0)
        if self.save_mistake_mips and rank == 0:
            util.mkdir(self.mistakes_dir, True)

        # Iterate over dataset
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(val_dataloader, desc="Validating", unit="batch", disable=rank != 0)
            for x, y in pbar:
                # Run model
                hat_y, loss = self.forward_pass(x, y)

                # Move to CPU
                y = ml_util.tensor_to_list(y)
                hat_y = ml_util.tensor_to_list(hat_y)

                # Store predictions
                y_accum.extend(y)
                hat_y_accum.extend(hat_y)
                loss_accum += float(ml_util.to_cpu(loss))
                if rank == 0:
                    pbar.set_postfix(loss=f"{loss_accum / len(y_accum):.4f}")

                # Save MIPs of mistakes (only rank 0 in distributed mode)
                if rank == 0:
                    self._save_mistake_mips(x, y, hat_y, idx_offset)
                idx_offset += len(y)

        # Write stats to tensorboard
        stats = self.compute_stats(y_accum, hat_y_accum)
        stats["loss"] = loss_accum / len(y_accum)
        self.update_tensorboard(stats, epoch, "val_")
        return stats

    def forward_pass(self, x, y):
        """
        Performs a forward pass through the model and computes loss.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (B, 2, D, H, W).
        y : torch.Tensor
            Ground truth labels with shape (B, 1).

        Returns
        -------
        hat_y : torch.Tensor
            Model predictions.
        loss : torch.Tensor
            Computed loss value.
        """
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            x = x.to(self.device)
            y = y.to(self.device)
            hat_y = self.model(x)
            loss = self.criterion(hat_y, y)
            return hat_y, loss

    # --- Helpers ---
    @staticmethod
    def compute_stats(y, hat_y):
        """
        Computes F1 score, precision, and recall for each sample in a batch.

        Parameters
        ----------
        y : torch.Tensor
            Ground truth labels with shape (B, 1).
        hat_y : torch.Tensor
            Predicted labels with shape (B, 1).

        Returns
        -------
        stats : Dict[str, float]
            Dictionary of metric names to values.
        """
        # Reformat predictions
        hat_y = (np.array(hat_y) > 0).astype(int)
        y = np.array(y, dtype=int)

        # Compute stats
        avg_prec = precision_score(y, hat_y, zero_division=np.nan)
        avg_recall = recall_score(y, hat_y, zero_division=np.nan)
        avg_f1 = 2 * avg_prec * avg_recall / max((avg_prec + avg_recall), 1)
        avg_acc = accuracy_score(y, hat_y)
        stats = {
            "f1": avg_f1,
            "precision": avg_prec,
            "recall": avg_recall,
            "accuracy": avg_acc
        }
        return stats

    @staticmethod
    def report_stats(stats, is_train=True):
        """
        Prints a summary of training or validation statistics.

        Parameters
        ----------
        stats : Dict[str, float]
            Dictionary of metric names to values.
        is_train : bool, optional
            Indication of whether stats were computed during training.
        """
        summary = "   Train: " if is_train else "   Val: "
        for key, value in stats.items():
            summary += f"{key}={value:.4f}, "
        print(summary)

    def check_model_performance(self, stats, epoch):
        """
        Checks whether the current model's performance (based on F1 score)
        surpasses the previous best, and saves the model if it does.

        Parameters
        ----------
        stats : Dict[str, float]
            Dictionary of evaluation metrics from the current epoch.
            Must contain the key "f1" representing the F1 score.
        epoch : int
            Current training epoch.

        Returns
        -------
        bool
            True if the model achieved a new best F1 score and was saved.
            False otherwise.
        """
        if stats["f1"] > self.best_f1 and stats["recall"] > self.min_recall:
            self.best_f1 = stats["f1"]
            self.save_model(epoch)
            return True
        else:
            return False

    def load_pretrained_weights(self, model_path):
        """
        Loads a pretrained model weights from a checkpoint file.

        Handles checkpoints saved with or without DDP 'module.' prefix,
        ensuring compatibility regardless of how the checkpoint was saved.

        Parameters
        ----------
        model_path : str
            Path to the checkpoint file containing the saved weights.
        """
        state_dict = torch.load(model_path, map_location=self.device)

        # Get the model to load into (handle DDP wrapping)
        model_to_load = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        # Check if state_dict has 'module.' prefix but model doesn't need it
        # or vice versa, and adjust accordingly
        model_keys = set(model_to_load.state_dict().keys())
        ckpt_keys = set(state_dict.keys())

        # If checkpoint has 'module.' prefix but model doesn't
        if any(k.startswith("module.") for k in ckpt_keys) and not any(
            k.startswith("module.") for k in model_keys
        ):
            state_dict = {
                k.replace("module.", "", 1): v for k, v in state_dict.items()
            }
        # If model expects 'module.' prefix but checkpoint doesn't have it
        elif any(k.startswith("module.") for k in model_keys) and not any(
            k.startswith("module.") for k in ckpt_keys
        ):
            state_dict = {f"module.{k}": v for k, v in state_dict.items()}

        model_to_load.load_state_dict(state_dict)

    def _save_mistake_mips(self, x, y, hat_y, idx_offset):
        """
        Saves MIPs of each false negative and false positive.

        Parameters
        ----------
        x : numpy.ndarray
            Input tensor with shape (B, 2, D, H, W).
        y : numpy.ndarray
            Ground truth labels with shape (B, 1).
        hat_y : numpy.ndarray
            Predicted labels with shape (B, 1).
        """
        if self.save_mistake_mips:
            # Initializations
            if isinstance(x, dict):
                x = ml_util.to_cpu(x["img"], True)
            else:
                x = ml_util.to_cpu(x, True)

            # Save MIPs
            for i, (y_i, hat_y_i) in enumerate(zip(y, hat_y)):
                mistake_type = classify_mistake(y_i, hat_y_i)
                if mistake_type:
                    filename = f"{mistake_type}{i + idx_offset}.png"
                    output_path = os.path.join(self.mistakes_dir, filename)
                    img_util.plot_image_and_segmentation_mips(
                        x[i, 0], 2 * x[i, 1], output_path
                    )

    def save_model(self, epoch):
        """
        Saves the current model state to a file.

        Handles DDP wrapping by extracting the underlying module before saving,
        ensuring checkpoints are compatible with both DDP and non-DDP loading.

        Parameters
        ----------
        epoch : int
            Current training epoch.
        """
        date = datetime.today().strftime("%Y%m%d")
        filename = f"{self.model_name}-{date}-{epoch}-{self.best_f1:.4f}.pth"
        path = os.path.join(self.log_dir, filename)

        # Unwrap DDP model if necessary for checkpoint compatibility
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        torch.save(model_to_save.state_dict(), path)

    def update_tensorboard(self, stats, epoch, prefix):
        """
        Logs scalar statistics to TensorBoard.

        Parameters
        ----------
        stats : Dict[str, float]
            Dictionary of metric names to lists of values.
        epoch : int
            Current training epoch.
        prefix : str
            Prefix to prepend to each metric name when logging.
        """
        if self.writer is None:
            return
        for key, value in stats.items():
            self.writer.add_scalar(prefix + key, stats[key], epoch)


class DistributedTrainer(Trainer):
    """
    A subclass of Trainer that uses DistributedDataParallel to train a model.
    """

    def _gather_predictions(self, y_local, hat_y_local):
        """
        Gather predictions from all ranks for accurate metric computation.

        Parameters
        ----------
        y_local : list
            Local ground truth labels.
        hat_y_local : list
            Local predictions.

        Returns
        -------
        tuple
            (y_gathered, hat_y_gathered) - combined lists from all ranks.
        """
        if not dist.is_initialized():
            return y_local, hat_y_local

        # Convert to tensors for gathering
        y_tensor = torch.tensor(y_local, dtype=torch.float32, device=self.device)
        hat_y_tensor = torch.tensor(
            hat_y_local, dtype=torch.float32, device=self.device
        )

        # Gather sizes from all ranks (they may differ)
        local_size = torch.tensor([len(y_local)], dtype=torch.float32, device=self.device)
        sizes = [torch.zeros(1, device=self.device) for _ in range(self.world_size)]
        dist.all_gather(sizes, local_size)
        sizes = [int(s.item()) for s in sizes]
        max_size = max(sizes)

        # Pad tensors to max size for gathering
        y_padded = torch.zeros(max_size, device=self.device)
        hat_y_padded = torch.zeros(max_size, device=self.device)
        y_padded[: len(y_local)] = y_tensor
        hat_y_padded[: len(hat_y_local)] = hat_y_tensor

        # Gather from all ranks
        y_gathered_list = [
            torch.zeros(max_size, device=self.device)
            for _ in range(self.world_size)
        ]
        hat_y_gathered_list = [
            torch.zeros(max_size, device=self.device)
            for _ in range(self.world_size)
        ]
        dist.all_gather(y_gathered_list, y_padded)
        dist.all_gather(hat_y_gathered_list, hat_y_padded)

        # Concatenate and trim to actual sizes
        y_gathered = []
        hat_y_gathered = []
        for i, size in enumerate(sizes):
            y_gathered.extend(y_gathered_list[i][:size].cpu().tolist())
            hat_y_gathered.extend(hat_y_gathered_list[i][:size].cpu().tolist())

        return y_gathered, hat_y_gathered

    def _gather_loss(self, loss_accum, n_samples_local):
        """
        Gather and properly normalize loss across all ranks.

        Parameters
        ----------
        loss_accum : float
            Accumulated loss on this rank.
        n_samples_local : int
            Number of samples processed on this rank.

        Returns
        -------
        float
            Properly normalized global loss.
        """
        if not dist.is_initialized():
            return loss_accum / n_samples_local

        # Gather total loss and sample counts from all ranks
        loss_tensor = torch.tensor([loss_accum], device=self.device)
        count_tensor = torch.tensor([n_samples_local], device=self.device)

        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

        return loss_tensor.item() / count_tensor.item()

    def __init__(
        self,
        model,
        model_name,
        output_dir,
        device="cuda",
        lr=1e-3,
        max_epochs=200,
        save_mistake_mips=False
    ):
        """
        Instantiates a DistributedTrainer object.

        Parameters
        ----------
        model : torch.nn.Module
            Model that is trained to perform binary classification.
        model_name : str
            Name of model used for logging and checkpointing.
        output_dir : str
            Directory that tensorboard and model checkpoints are written to.
        lr : float, optional
            Learning rate. Default is 1e-3.
        max_epochs : int, optional
            Maximum number of training epochs. Default is 200.
        """
        # Check that multiple GPUs are available
        msg = "Error: only a single GPU detected in environment!"
        assert "RANK" in os.environ and "WORLD_SIZE" in os.environ, msg

        # Get rank info before calling parent (needed for directory creation)
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])

        # Initialize process group first so we can use barriers
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=self.world_size,
            rank=self.rank,
        )

        # Generate consistent session name across all ranks
        # Rank 0 generates the name and broadcasts to others
        if self.rank == 0:
            exp_name = "session-" + datetime.today().strftime("%Y%m%d_%H%M")
            exp_name_tensor = torch.tensor(
                [ord(c) for c in exp_name] + [0] * (64 - len(exp_name)),
                dtype=torch.int64,
                device=f"cuda:{self.local_rank}"
            )
        else:
            exp_name_tensor = torch.zeros(64, dtype=torch.int64, device=f"cuda:{self.local_rank}")

        dist.broadcast(exp_name_tensor, src=0)
        exp_name = "".join(chr(c) for c in exp_name_tensor.tolist() if c != 0)
        log_dir = os.path.join(output_dir, exp_name)
        mistakes_dir = os.path.join(log_dir, "mistakes")

        # Only rank 0 creates directories
        if self.rank == 0:
            util.mkdir(log_dir)
            if save_mistake_mips:
                util.mkdir(mistakes_dir)

        # Barrier to ensure directories are created before other ranks proceed
        dist.barrier()

        # Now initialize parent class attributes without creating directories
        self.best_f1 = 0
        self.device = device
        self.log_dir = log_dir
        self.max_epochs = max_epochs
        self.min_recall = 0
        self.mistakes_dir = mistakes_dir
        self.model_name = model_name
        self.save_mistake_mips = save_mistake_mips

        self.criterion = nn.BCEWithLogitsLoss()
        self.model = model.to(device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)

        # Only rank 0 writes to TensorBoard to avoid file contention
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

        # Store lr for optimizer creation after DDP setup
        self._lr = lr

        # Set up DDP (process group already initialized)
        # Optimizer and scheduler are created after DDP wrapping
        self._setup_ddp()

    def _setup_ddp(self):
        """
        Wrap model in DDP and create optimizer/scheduler.

        The optimizer must be created after DDP wrapping to ensure it
        references the correct parameters for gradient synchronization.
        """
        # Set device
        self.device = torch.device(f"cuda:{self.local_rank}")

        # Move model to local rank device
        self.model = self.model.to(self.device)
        self.set_contiguous_model()

        # Wrap model in DDP
        self.model = DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=False,
        )

        # Create optimizer and scheduler AFTER DDP wrapping
        # Only pass trainable parameters to optimizer
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self._lr,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_epochs)

        print(f"Initialized rank {self.rank}/{self.world_size}")

    def set_contiguous_model(self):
        """
        Ensures all model parameters are stored in contiguous memory.
        """
        for p in self.model.parameters():
            if not p.is_contiguous():
                p.data = p.contiguous()

    def validate_step(self, val_dataloader, epoch):
        """
        Performs a full validation loop with proper metric aggregation across
        all ranks.

        Parameters
        ----------
        val_dataloader : torch.utils.data.DataLoader
            DataLoader for the validation dataset.
        epoch : int
            Current training epoch.

        Returns
        -------
        stats : Dict[str, float]
            Dictionary of aggregated validation metrics.
        """
        # Initializations
        idx_offset = 0
        loss_accum = 0
        y_accum = list()
        hat_y_accum = list()
        if self.save_mistake_mips and self.rank == 0:
            util.mkdir(self.mistakes_dir, True)

        # Iterate over dataset
        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(val_dataloader, desc="Validating", unit="batch", disable=self.rank != 0)
            for x, y in pbar:
                # Run model
                hat_y, loss = self.forward_pass(x, y)

                # Move to CPU
                y = ml_util.tensor_to_list(y)
                hat_y = ml_util.tensor_to_list(hat_y)

                # Store predictions
                y_accum.extend(y)
                hat_y_accum.extend(hat_y)
                loss_accum += float(ml_util.to_cpu(loss))
                if self.rank == 0:
                    pbar.set_postfix(loss=f"{loss_accum / len(y_accum):.4f}")

                # Save MIPs of mistakes (only rank 0)
                if self.rank == 0:
                    self._save_mistake_mips(x, y, hat_y, idx_offset)
                idx_offset += len(y)

        # Gather predictions from all ranks for accurate metrics
        y_gathered, hat_y_gathered = self._gather_predictions(y_accum, hat_y_accum)

        # Compute properly normalized loss across all ranks
        global_loss = self._gather_loss(loss_accum, len(y_accum))

        # Compute stats on gathered predictions (only meaningful on rank 0,
        # but all ranks compute to stay synchronized)
        stats = self.compute_stats(y_gathered, hat_y_gathered)
        stats["loss"] = global_loss

        # Only rank 0 writes to tensorboard
        if self.rank == 0:
            self.update_tensorboard(stats, epoch, "val_")

        return stats

    def run(self, train_dataloader, val_dataloader):
        """
        Runs the full training and validation loop.
        Works for both single-GPU and DDP.

        Parameters
        ----------
        train_dataloader : torch.utils.data.DataLoader
            Dataloader used for training.
        val_dataloader : torch.utils.data.DataLoader
            Dataloader used for validation.
        """
        # Report experiment name
        rank = getattr(self, "rank", 0)
        if rank == 0:
            exp_name = os.path.basename(os.path.normpath(self.log_dir))
            print("\nExperiment:", exp_name)

        # Main
        for epoch in range(self.max_epochs):
            # Set epoch
            train_sampler = getattr(train_dataloader, "sampler", None)
            if isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(epoch)

            # Train-Validate
            train_stats = self.train_step(train_dataloader, epoch)
            val_stats = self.validate_step(val_dataloader, epoch)

            # Synchronize before model saving to ensure all ranks have
            # completed validation
            dist.barrier()

            # Report results and save model (rank 0 only)
            if rank == 0:
                new_best = self.check_model_performance(val_stats, epoch)
                print(f"\nEpoch {epoch}: ", "New Best!" if new_best else "")
                self.report_stats(train_stats, is_train=True)
                self.report_stats(val_stats, is_train=False)

            # Barrier after saving to ensure checkpoint is written before
            # proceeding to next epoch
            dist.barrier()

            # Step scheduler
            self.scheduler.step()


# --- Helpers ---
def classify_mistake(y_i, hat_y_i):
    """
    Classify a prediction mistake for a single example.

    Parameters
    ----------
    y_i : int
        Ground truth label.
    hat_y_i : float
        Predicted label.

    Returns
    -------
    str or None
        Name of mistake or None if prediction is correct.
    """
    if y_i == 1 and hat_y_i < 0:
        return "false_negative"
    if y_i == 0 and hat_y_i > 0:
        return "false_positive"
    return None
