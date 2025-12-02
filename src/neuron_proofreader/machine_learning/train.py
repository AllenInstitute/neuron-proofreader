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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

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
    batch_size : int
        Number of samples per batch during training.
    best_f1 : float
        Best F1 score achieved so far on valiation dataset.
    criterion : torch.nn.BCEWithLogitsLoss
        Loss function used during training.
    device : str, optional
        Device that model is run on. Default is "cuda".
    log_dir : str
        Path to directory that tensorboard and checkpoints are saved to.
    max_epochs : int
        Maximum number of training epochs.
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
        batch_size=32,
        device="cuda",
        lr=1e-3,
        max_epochs=200,
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
        batch_size : int, optional
            Number of samples per batch during training. Default is 32.
        lr : float, optional
            Learning rate. Default is 1e-3.
        max_epochs : int, optional
            Maximum number of training epochs. Default is 200.
        save_mistake_mips : bool, optional
            Indication of whether to save MIPs of mistakes. Default is False.
        """
        # Set experiment name
        exp_name = "session-" + datetime.today().strftime("%Y%m%d_%H%M")
        log_dir = os.path.join(output_dir, exp_name)
        util.mkdir(log_dir)

        # Instance attributes
        self.batch_size = batch_size
        self.best_f1 = 0
        self.device = device
        self.log_dir = log_dir
        self.max_epochs = max_epochs
        self.mistakes_dir = os.path.join(log_dir, "mistakes")
        self.model_name = model_name
        self.save_mistake_mips = save_mistake_mips

        self.criterion = nn.BCEWithLogitsLoss()
        self.model = model.to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=25)
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
        for x_i, y_i in train_dataloader:
            # Forward pass
            self.optimizer.zero_grad()
            hat_y_i, loss_i = self.forward_pass(x_i, y_i)

            # Backward pass
            self.scaler.scale(loss_i).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Store results
            y.extend(ml_util.to_cpu(y_i, True).flatten().tolist())
            hat_y.extend(ml_util.to_cpu(hat_y_i, True).flatten().tolist())
            loss.append(float(ml_util.to_cpu(loss_i)))

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
        util.mkdir(self.mistakes_dir, True)
        loss_list, y_list, hat_y_list = list(), list(), list()
        self.model.eval()
        with torch.no_grad():
            for x, y in val_dataloader:
                # Run model
                hat_y, loss = self.forward_pass(x, y)

                # Move to CPU
                y = ml_util.to_cpu(y, True).flatten().tolist()
                hat_y = ml_util.to_cpu(hat_y, True).flatten().tolist()

                # Store predictions
                y_list.extend(y)
                hat_y_list.extend(hat_y)
                loss_list.append(float(ml_util.to_cpu(loss)))
                self._save_mistake_mips(x, y, hat_y)

        # Write stats to tensorboard
        stats = self.compute_stats(y_list, hat_y_list)
        stats["loss"] = np.mean(loss_list)
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
            Ground truth labels of shape (B, 1).
        hat_y : torch.Tensor
            Model predictions of the same shape as ground truth.

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
        if stats["f1"] > self.best_f1:
            self.best_f1 = stats["f1"]
            self.save_model(epoch)
            return True
        else:
            return False

    def load_pretrained_weights(self, model_path):
        """
        Loads a pretrained model weights from a checkpoint file.

        Parameters
        ----------
        model_path : str
            Path to the checkpoint file containing the saved weights.
        """
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )

    def _save_mistake_mips(self, x, y, hat_y):
        if self.save_mistake_mips:
            # Initializations
            if isinstance(x, dict):
                x = ml_util.to_cpu(x["img"], True)
            else:
                x = ml_util.to_cpu(x, True)

            # Save MIPs
            for i, (y_i, hat_y_i) in enumerate(zip(y, hat_y)):
                # Check for mistake
                if y_i == 1 and hat_y_i < 0:
                    name = "false_negative"
                elif y_i == 0 and hat_y_i > 0:
                    name = "false_positive"
                else:
                    name = None

                # Save MIP
                if name:
                    cnt = len(util.listdir(self.mistakes_dir, ".png")) + 1
                    output_path = os.path.join(self.mistakes_dir, f"{name}{cnt}.png")
                    img_util.plot_image_and_segmentation_mips(
                        x[i, 0, ...], x[i, 1, ...], output_path
                    )

    def save_model(self, epoch):
        """
        Saves the current model state to a file.

        Parameters
        ----------
        epoch : int
            Current training epoch.
        """
        date = datetime.today().strftime("%Y%m%d")
        filename = f"{self.model_name}-{date}-{epoch}-{self.best_f1:.4f}.pth"
        path = os.path.join(self.log_dir, filename)
        torch.save(self.model.state_dict(), path)

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
        for key, value in stats.items():
            self.writer.add_scalar(prefix + key, stats[key], epoch)


class DistributedTrainer(Trainer):
    """
    A subclass of Trainer that uses DistributedDataParallel to train a model.
    """

    def __init__(
        self,
        model,
        model_name,
        output_dir,
        batch_size=32,
        device="cuda",
        lr=1e-3,
        max_epochs=200,
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
        batch_size : int, optional
            Number of samples per batch during training. Default is 32.
        lr : float, optional
            Learning rate. Default is 1e-3.
        max_epochs : int, optional
            Maximum number of training epochs. Default is 200.
        """
        # Call parent class
        super().__init__(
            model,
            model_name,
            output_dir,
            batch_size=batch_size,
            device=device,
            lr=lr,
            max_epochs=max_epochs,
        )

        # Check that multiple GPUs are available
        msg = "Error: only a single GPU detected in environment!"
        assert "RANK" in os.environ and "WORLD_SIZE" in os.environ, msg

        # Instance attributes
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])

        # Set up DDP
        self._setup_ddp()

    def _setup_ddp(self):
        """
        Initialize torch.distributed and wrap model in DDP.
        """
        # Set device
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")

        # Initialize process group
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=self.world_size,
            rank=self.rank,
        )

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
        print(f"Initialized rank {self.rank}/{self.world_size}")

    def set_contiguous_model(self):
        """
        Ensures all model parameters are stored in contiguous memory.
        """
        for p in self.model.parameters():
            if not p.is_contiguous():
                p.data = p.contiguous()

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

            # Report results
            if rank == 0:
                new_best = self.check_model_performance(val_stats, epoch)
                print(f"\nEpoch {epoch}: ", "New Best!" if new_best else "")
                self.report_stats(train_stats, is_train=True)
                self.report_stats(val_stats, is_train=False)

            # Step scheduler
            self.scheduler.step()
