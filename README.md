# DiffuBot

This repository contains code and scripts for training and evaluating a model using the Pix2Pix framework.

## Files Overview

- **`commit.py`**: Commits dataset onto Hugging Face.
- **`data_process.py`**: Processes head camera data from `.pkl` files by sampling, resizing, and normalizing RGB images, then saves the data in both `.pkl` and JSON formats. The final JSON dataset is pushed to Hugging Face for use in training a model, with each entry containing "before" and "after" images along with a corresponding prediction prompt.
- **`evaluate_final.ipynb`**: Computes evaluation criteria (SSIM, PSNR) and evaluate the trained model on tesing images.
- **`inference_p2p.py`**: Performs inference on the dataset using a fine-tuned InstructPix2Pix model, generating predicted images based on input prompts and "before" images. The results, including "before", "predicted", and "actual" images, are saved to disk and visualized in comparison plots for evaluation.
- **`main.py`**: Main training script for the model.
- **`pre_processing.py`**: Processes head camera data from `.pkl` files by extracting, sampling, resizing, and normalizing RGB images, then saves the processed data into a new `.pkl` file. The final data is organized by subdataset and episode, with each sample containing resized RGB images and camera parameters.

For a more detailed explanation, please refer to the corresponding files.

## Installation

To install the required Python libraries, run the command `pip install -r requirements.txt` in the terminal.

## Dataset

### Data Generation

Data generation mainly relies on simulation. For specific details, please refer to the [RoboTwin project repository](https://github.com/TianxingChen/RoboTwin/tree/main). This repository provides a complete set of simulation environment setup, data collection processes, and related code, helping you understand the entire data generation process.

### Data Preprocessing

Data preprocessing is completed by running the `data_process.py` script. This script covers key operations such as data cleaning, format conversion and so on.

### Data Download

You can directly download the processed data from the Hugging Face platform. The access path is [Aurora1609/RoboTwin](https://huggingface.co/datasets/Aurora1609/RoboTwin). After downloading, store it according to the default directory structure, and it can be directly used for subsequent research and experiments.

## Benchmark

We used the SSIM (Structural Similarity Index) and PSNR (Peak Signal-to-Noise Ratio) as the evaluation metrics.

## Experiment and Result

Train our Fine-Tuned model through `main.py`, generating predicted pictures using `inference_p2p.py`, then compare the model output with ground truth by `evaluate_final.ipynb`. The final evaluation results of the baseline and fine-tuned models are summarized in the table below.

| **Model**   | **Mean SSIM** | **Mean PSNR** |
|-------------|---------------|---------------|
| Baseline    | 0.6161        | 11.7397       |
| Fine-Tuned  | 0.8629        | 21.2033       |

More figures and analysis can be found in our report.
