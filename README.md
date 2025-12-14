# Video Summarization with Large Language Models (LLMVS) - Project Implementation

This repository contains the implementation and analysis code for the **LLMVS** project, based on the CVPR 2025 paper by Lee et al.

## 📂 Implementation Files

The core logic and analysis scripts for this project are located in this repository:

*   **`LLMVS_Implementation_TVSum.ipynb`**:
    *   The main **Google Colab Notebook** containing the full run results, including model loading, scoring, and evaluation.
    *   Use this notebook to reproduce our experiments on the TVSum dataset.

*   **`create_summary_video.py`**:
    *   A script to generate the final summary video (`.mp4`) by stitching together selected frames based on the model's importance scores.

*   **`save_frame_indices.py`**:
    *   Extracts and saves the frame indices and their corresponding importance scores for analysis.

*   **`llmvs_implementation_tvsum.py`**:
    *   Python script version of the implementation logic for the TVSum dataset.

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
To evaluate the trained model on the test split:

```bash
python test_splits.py --dataset tvsum --weights rho
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
python create_summary_video.py \
    --json_path tvsum_frame_indices_with_scores.json \
    --h5_path TVSum/eccv16_dataset_tvsum_google_pool5.h5 \
    --video_dir ./datasets/tvsum50_ver_1_1/ydata-tvsum50-v1_1/video \
    --video_key video_45 \
    --output video_45_summary.mp4
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
