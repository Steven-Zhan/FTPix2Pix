# DiffuBot

This repository contains code and scripts for training and evaluating a model using the Pix2Pix framework.

## Files Overview

- **`commit.py`**: Commits dataset onto Hugging Face.
- **`evaluate_final.ipynb`**: Computes evaluation criteria (SSIM, PSNR) and evaluate the trained model on tesing images.
- **`inference_p2p.py`**: Performs inference on the dataset using a fine-tuned InstructPix2Pix model, generating predicted images based on input prompts and "before" images. The results, including "before", "predicted", and "actual" images, are saved to disk and visualized in comparison plots for evaluation.
- **`pre_processing.py`**: Processes head camera data from `.pkl` files by extracting, sampling, resizing, and normalizing RGB images, then saves the processed data into a new `.pkl` file. The final data is organized by subdataset and episode, with each sample containing resized RGB images and camera parameters.
- **`process.py`**: Processes head camera data from `.pkl` files by sampling, resizing, and normalizing RGB images, then saves the data in both `.pkl` and JSON formats. The final JSON dataset is pushed to Hugging Face for use in training a model, with each entry containing "before" and "after" images along with a corresponding prediction prompt.
- **`refined_train_script.py`**: Main training script for the model.

For a more detailed explanation, please refer to the corresponding files.

## GitHub Repository

You can access the repository here:  
[https://github.com/Steven-Zhan/FTPix2Pix](https://github.com/Steven-Zhan/FTPix2Pix)
