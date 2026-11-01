# Neuron Proofreader

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Python](https://img.shields.io/badge/python->=3.7-blue?logo=python)

**Neuron Proofreader** automatically corrects errors in large-scale neuron reconstructions. Given a set of neuron fragments as SWC files and the image they were traced from, it fixes two kinds of mistakes:

- **Splits** — one neuron broken into several fragments, which are reconnected.
- **Merges** — two neurons joined into one fragment, which are cut apart.

Both tasks are handled by a single model that combines a 3D CNN on image patches with an [Arborist](https://github.com/AllenInstitute/arborist) encoder on the local skeleton geometry.

<p align="center">
  <img src="imgs/result.png" width="900" alt="Fragments before and after proofreading">
  <br>
  <b>Figure:</b> fragmented neuron segments reconnected into coherent traces.
</p>

## How it works

Fragments are loaded into a graph, then proofread in five steps:

1. **Learned merge detection** — a CNN scores sites along every fragment and splits the fragment at predicted merge errors.
2. **Soma-based merge detection** — fragments that touch more than one soma are split between them.
3. **Soma split correction** — fragments ending within a short distance of a soma are connected to it.
4. **Learned split correction** — proposals (candidate connections between nearby fragment endpoints) are generated at increasing search radii and scored by a graph neural network; accepted proposals become edges. A final long-range pass searches a narrow cone along each straight fragment's tangent.
5. **Final merge detection** — the corrected graph is rescored for merges, for QC only.

<p align="center">
  <img src="imgs/pipeline.png" width="800" alt="Split correction pipeline">
  <br>
  <b>Figure:</b> split correction pipeline.
</p>

## Installation

```bash
git clone https://github.com/AllenInstitute/arborist.git
pip install -e arborist

git clone https://github.com/AllenInstitute/neuron-proofreader.git
pip install -e neuron-proofreader
```

A CUDA GPU is strongly recommended for inference.

## Usage

```python
from neuron_proofreader import configs
from neuron_proofreader.models.joint_model import JointProofreader
from neuron_proofreader.proofreading_pipeline import ProofreadPipeline

model = JointProofreader.load("model.pth", map_location="cuda").to("cuda").eval()

graph_config = configs.GraphConfig(anisotropy=(0.748, 0.748, 1.0), node_spacing=5)
img_config = configs.ImageConfig(img_path="s3://bucket/brain/image.zarr", patch_shape=(96, 96, 96))
merge_config = configs.MergeInferenceConfig(modality="image_graph", patch_shape=(96, 96, 96), batch_size=32)
split_config = configs.SplitInferenceConfig(patch_shape=(96, 96, 96), batch_size=32)
proposals_config = configs.ProposalsConfig(min_search_radius=10, max_search_radius=30, delta_search_radius=10)

pipeline = ProofreadPipeline("fragments/", graph_config, img_config, "results/", soma_centroids=somas)
pipeline.learned_merge_detection(model.as_merge_model(), merge_config)
pipeline.merge_proofreading("connected_somas")
pipeline.soma_split_proofreading()
pipeline.learned_split_detection(model.as_split_model(), proposals_config, split_config)
pipeline.save_final_result()
```

Each step writes its corrected SWCs and a log to a subdirectory of `results/`.

## Repository layout

| Package | Contents |
|---|---|
| `proofreading_pipeline.py` | `ProofreadPipeline`, the end-to-end driver |
| `proposal_graph.py`, `fragments_graph.py` | Graph of fragments and the proposals between them |
| `split_proofreading/` | Proposal generation, feature extraction and GNN inference for splits |
| `merge_proofreading/` | Site search, inference and training data for merges |
| `models/` | `JointProofreader` and its vision / skeleton components |
| `machine_learning/` | Image loading, augmentation, subgraph sampling, training loop |
| `configs.py` | Dataclasses for every tunable parameter |

## Contact

Anna Grim — anna.grim@alleninstitute.org

## License

MIT. See [LICENSE](LICENSE).
