# Video Summarization with Large Language Models (LLMVS) - Project Implementation

This repository contains the implementation and analysis code for the **LLMVS** project, based on the CVPR 2025 paper by Lee et al.

## 📂 Repository Structure

### 1. Core Model & Training
Referenced from the original repository:

*   **`train.py`**: Main training script.
*   **`test.py`**: Evaluation script.
*   **`networks/`**: Model architecture definitions.
*   **`utils/`**: Helper functions for data loading and configuration.

### 2. Our Implementation & Extensions
The core logic and analysis scripts developed for this project. We adapted the codebase to utilize **PyTorch Lightning (>=2.0.0)** and integrated libraries compatible with the **Google Colab** environment (e.g., `torchmetrics`, `einops`, `h5py`).

*   **`LLMVS_Implementation_TVSum.ipynb`**:
    *   The main **Google Colab Notebook** containing the full run results, including model loading, scoring, and evaluation.
    *   Use this notebook to reproduce our experiments on the TVSum dataset.

*   **`create_summary_video.py`**:
    *   A script to generate the final summary video (`.mp4`) by stitching together selected frames based on the model's importance scores.

*   **`save_frame_indices.py`**:
    *   Extracts and saves the frame indices and their corresponding importance scores for analysis.

*   **`llmvs_implementation_tvsum.py`**:
    *   **Raw Python export** of the Colab notebook.
    *   **Note**: This file is provided for reference. It contains Colab-specific commands and hardcoded paths that must be modified to run locally.

*   **`analyze_metrics.py`**:
    *   Analyzes evaluation results (`eval_metrics.csv`) to generate summary statistics and visualization plots (`per_video_metrics.png`, `average_metrics.png`).

*   **`compare_sumamaries.py`**:
    *   Compares two text summaries (e.g., original vs. generated) using ROUGE, BERTScore, and Cosine Similarity metrics.

*   **`Evaluation/`**:
    *   Directory containing evaluation metrics and results.

## 🎥 Results & Demos

*   **Run Results**: See the [Jupyter Notebook](LLMVS_Implementation_TVSum.ipynb) for detailed execution logs and score visualizations.
*   **Video Summaries**: View the generated summary videos and original test videos on Google Drive:
    *   **[📂 View Results on Google Drive](https://drive.google.com/drive/folders/1J1dMqF1lrqOlGkA_rVMLwDcIz4K8MW7v?usp=drive_link)**

## ⚠️ Implementation Notes & Code Review

The file `llmvs_implementation_tvsum.py` is a Python export of our Colab notebook. If running this locally, please note the following:

1.  **Hardcoded Paths**:
    *   The script contains absolute paths specific to our local machine (e.g., `/Users/konda/...`).
    *   **Action**: You must update `--video_dir` and other paths to match your local directory structure before running.
2.  **Colab Dependencies**:
    *   Commands like `drive.mount` and `!pip install` are specific to Google Colab.
    *   **Action**: Comment these out if running in a standard local Python environment.
3.  **Checkpoint Names**:
    *   The script references specific checkpoint files (e.g., `epoch=143...`).
    *   **Action**: Update these filenames to match the actual checkpoints generated during your training run.

## 🚀 Training & Testing Commands

Below are the exact commands used in our implementation to train and evaluate the model on the TVSum dataset.

### 1. Training
We trained the model for 200 epochs using the following configuration:

```bash
python train.py \
    --tag tvsum_split0 \
    --model tvsum_head2_layer3 \
    --lr 0.00007 \
    --epochs 200 \
    --dataset tvsum \
    --reduced_dim 2048 \
    --num_heads 2 \
    --num_layers 3 \
    --split_idx 0 \
    --pt_path 'llama_emb/tvsum_sum/'
```

### 2. Evaluation
Since we only trained on Split 0, we evaluate specifically on that split:

```bash
python test.py \
    --dataset tvsum \
    --split_idx 0 \
    --weights Summaries/best_rho_model/YOUR_BEST_CHECKPOINT.ckpt \
    --pt_path llama_emb/tvsum_sum/
```

### 3. Generating Results
To extract frame scores and generate the summary video:

```bash
# Save frame scores to JSON
python save_frame_indices.py \
    --weights Summaries/tvsum_head2_layer3/tvsum/tvsum_split0/best_rho_model/YOUR_BEST_CHECKPOINT.ckpt \
    --pt_path llama_emb/tvsum_sum/ \
    --dataset tvsum \
    --split_idx 0 \
    --model tvsum_head2_layer3 \
    --tag tvsum_split0 \
    --output_json tvsum_frame_indices_with_scores.json

# Generate Video Summary
# Note: --video_dir must point to the folder containing the raw video files (e.g., .mp4)
python create_summary_video.py \
    --json_path tvsum_frame_indices_with_scores.json \
    --h5_path TVSum/eccv16_dataset_tvsum_google_pool5.h5 \
    --video_dir /path/to/your/raw/videos \
    --video_key video_45 \
    --output video_45_summary.mp4
```

### 4. Analysis Tools

**Analyze Evaluation Metrics**:
Generates plots and statistics from `eval_metrics.csv`.
```bash
python analyze_metrics.py
```

**Compare Summaries**:
Calculates semantic similarity between two text summaries.
```bash
python compare_sumamaries.py \
    --long_summary Evaluation/long_summary_14.txt \
    --short_summary Evaluation/short_summary_14.txt
```

## 📄 Original Work Citation

This project builds upon the official implementation of:

**Video Summarization with Large Language Models**
*Min Jung Lee, Dayoung Gong, Minsu Cho*
CVPR 2025

*   **Original Repository**: [https://github.com/mlee47/LLMVS](https://github.com/mlee47/LLMVS)
*   **Project Page**: [https://postech-cvlab.github.io/LLMVS/](https://postech-cvlab.github.io/LLMVS/)

```bibtex
@inproceedings{lee2025video,
  title={Video Summarization with Large Language Models},
  author={Lee, Min Jung and Gong, Dayoung and Cho, Minsu},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={18981--18991},
  year={2025}
}
```
