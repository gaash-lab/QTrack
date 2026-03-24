<!-- full-width, left-aligned -->
<div align="left">
  <img src="https://user-images.githubusercontent.com/74038190/212284115-f47cd8ff-2ffb-4b04-b5bf-4d1c14c0247f.gif"
       width="100%" />
</div>

<div align="center">

<table>
<tr>
<td align="center">
<img src="https://github.com/user-attachments/assets/8a25fe07-06bb-4701-ba0a-88109b9be929" width="350" height="330">
</td>

<td align="center">
<img src="https://github.com/user-attachments/assets/2891bb81-f3f0-4d01-8323-a4cf1ac9c68b" width="350" height="350">
</td>
</tr>
</table>




# [QTrack: Query-Driven Reasoning for Multi-modal MOT](https://arxiv.org/abs/2603.13759)
[![Website](https://img.shields.io/badge/website-QTrack-purple)](https://gaash-lab.github.io/QTrack/)
[![Weights](https://img.shields.io/badge/weights-huggingface-blue)](https://huggingface.co/GAASH-Lab/QTrack)
[![Arxiv](https://img.shields.io/badge/Arxiv-paper-red?style=plastic&logo=arxiv)](https://arxiv.org/abs/2603.13759)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

This is the official PyTorch implementation of QTrack:

["**QTrack: Query-Driven Reasoning for Multi-modal MOT**"](https://arxiv.org/abs/2603.13759) by [Tajamul Ashraf](https://www.tajamulashraf.com/), [Tavaheed Tariq](https://tavaheed.netlify.app/), [Sonia Yadav](https://sonia-yadav.netlify.app/), [Abrar Ul Riyaz](https://abrarulriyaz.vercel.app/), [Wasif Tak](), [Moloud Abdar](), and [Janibul Bashir](https://www.janibbashir.com/).

---

## NEWS
- **[03/25/2026]** :collision: QTrack achieves new state-of-the-art on RMOT26 benchmark with **0.30 MCP** and **0.75 MOTP**! Check out our [project page](https://gaash-lab.github.io/QTrack/) for demos.
- **[03/18/2026]** We released the RMOT26 benchmark and QTrack codebase. See more details in our [arXiv paper](https://arxiv.org/abs/2603.13759)!
- **[03/10/2026]** Dataset and model checkpoints are now publicly available.

---

<div align="center">
<img width="1149" height="745" alt="image" src="https://github.com/user-attachments/assets/1ee0afa6-23e5-498e-851b-b1c3669d41d8" />

<p><em>QTrack performs query-driven multi-object tracking based on natural language instructions, tracking only the specified targets while maintaining temporal coherence.</em></p>
</div>

## 🎯 What is QTrack?

**Multi-object tracking (MOT)** has traditionally focused on estimating trajectories of all objects in a video, without selectively reasoning about user-specified targets under semantic instructions. In this work, we introduce a **query-driven tracking paradigm** that formulates tracking as a spatiotemporal reasoning problem conditioned on natural language queries. Given a reference frame, a video sequence, and a textual query, the goal is to localize and track only the target(s) specified in the query while maintaining temporal coherence and identity consistency.

**Key Contributions:**
- **RMOT26 Benchmark**: A large-scale benchmark with grounded queries and sequence-level splits to prevent identity leakage and enable robust generalization evaluation
- **QTrack Model**: An end-to-end vision-language model that integrates multi-modal reasoning with tracking-oriented localization
- **Temporal Perception-Aware Policy Optimization (TPA-PO)**: A structured reward strategy to encourage motion-aware reasoning

🔥 Check out our [project website](https://gaash-lab.github.io/QTrack/) for more overview and demos!

---

## 📊 Benchmark Results

QTrack achieves state-of-the-art performance on the [RMOT26](https://huggingface.co/datasets/GAASH-Lab/RMOT26) benchmark, significantly outperforming both open-source and closed-source models.

### Main Results on RMOT26

| Model | Params | MCP↑ | MOTP↑ | CLE (px)↓ | NDE↓ |
|:-----:|:------:|:----:|:-----:|:---------:|:----:|
| GPT-5.2 | - | 0.25 | 0.61 | 94.2 | 0.55 |
| Qwen3-VL-Instruct | 8B | 0.25 | 0.64 | 96.0 | 0.97 |
| Gemma 3 | 27B | 0.24 | 0.56 | 58.4 | 0.88 |
| Gemma 3 | 12B | 0.18 | 0.73 | 172.9 | 0.95 |
| VisionReasoner | 7B | 0.23 | 0.24 | 428.9 | 2.24 |
| Qwen2.5-VL-Instruct | 7B | 0.24 | 0.48 | 289.2 | 2.07 |
| InternVL | 8B | 0.21 | 0.66 | 117.44 | 0.64 |
| gpt-4o-mini | - | 0.20 | 0.57 | 130.48 | 0.67 |
| **QTrack (Ours)** | **3B** | **0.30** | **0.75** | **44.61** | **0.39** |

### Comparison with Traditional MOT Methods

| | **MOT17 Dataset** | | | | **DanceTrack Dataset** | | | |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Model** | **MOTA** | **MOTP** | **HOTA** | **MCP** | **MOTA** | **MOTP** | **HOTA** | **MCP** |
| MOTR | 0.61 | 0.81 | 0.22 | 0.44 | 0.42 | 0.70 | 0.35 | 0.51 |
| BoostTrack++ | 0.63 | 0.76 | 0.38 | 0.44 | - | - | - | - |
| MOTRv2 | - | - | - | - | 0.49 | 0.73 | 0.37 | 0.52 |
| TrackTrack | 0.75 | 0.50 | 0.23 | 0.29 | 0.36 | 0.73 | 0.40 | 0.55 |
| VisionReasoner | 0.64 | 0.86 | 0.60 | 0.21 | 0.59 | 0.85 | 0.61 | 0.26 |
| **QTrack (Ours)** | **0.69** | **0.87** | **0.69** | 0.26 | **0.63** | **0.83** | **0.66** | 0.35 |

### Fine-tuned VLLM Comparison

| Model | Params | MCP↑ | MOTP↑ | MOTA↑ | NDE↓ |
|:-----:|:------:|:----:|:-----:|:-----:|:----:|
| VisionReasoner | 3B | 0.22 | 0.65 | 0.01 | 0.76 |
| Gemma3 | 4B | 0.18 | 0.73 | -0.16 | 0.95 |
| Qwen2.5-VL | 3B | 0.14 | 0.76 | -0.51 | 3.41 |
| **QTrack (Ours)** | **3B** | **0.30** | **0.75** | **0.21** | **0.39** |

---

## 🔧 Installation

### Requirements
- Python ≥ 3.12
- PyTorch ≥ 2.6
- CUDA ≥ 12.1
- Transformers ≥ 4.51.3

### Setup Environment

```bash
# Create conda environment
conda create -n qtrack python=3.12
conda activate qtrack

# Install PyTorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install QTrack and dependencies
git clone https://github.com/gaash-lab/QTrack.git
cd QTrack
pip install -r requirements.txt
pip install -e .
```

## Training
Please change model path and data paths in [config file](training_scripts/visionreasoner_7b.yaml) and [bash script](training_scripts/run_visionreasoner_7b_4x80G.sh).

```bash
export VLLM_USE_MODEL_INSPECTOR=0
export VLLM_ATTENTION_BACKEND=XFORMERS

cd /home/gaash/Wasif/Tawheed/Seg-Zero_with_TAPO/training_scripts
bash run_visionreasoner_7b_4x80G.sh
```

---

## Evaluation

```bash
python evaluation_scripts/common_evaluation_visionreasoner_prev.py \
  --json <test json file> \
  --dataset_root <data est dir> \
  --model_path <model path> \
  --out <output json path>"
```
---

## Citation
If you find QTrack useful for your research, please cite:

```bibtex
@misc{ashraf2026qtrackquerydrivenreasoningmultimodal,
      title={QTrack: Query-Driven Reasoning for Multi-modal MOT}, 
      author={Tajamul Ashraf and Tavaheed Tariq and Sonia Yadav and Abrar Ul Riyaz and Wasif Tak and Moloud Abdar and Janibul Bashir},
      year={2026},
      eprint={2603.13759},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.13759}, 
}
```
