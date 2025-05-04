#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to fine-tune Stable Diffusion for InstructPix2Pix."""

import argparse
import logging
import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import datasets
import numpy as np
import PIL
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.constants import DIFFUSERS_REQUEST_TIMEOUT
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module


# Check wandb and import if available
if is_wandb_available():
    import wandb

# Check diffusers version
check_min_version("0.34.0.dev0")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = get_logger(__name__, log_level="INFO")

# Dataset mapping and wandb log columns
MAPPINGS_DATA = {
    "fusing/instructpix2pix-1000-samples": ("input_image", "edit_prompt", "edited_image"),
}
WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt"]

# Validation logging
def validate_log(pips, config, generator=None):
    logger.info(
        f"Validating... Generating {config.num_validation_images} images "
        f"for prompt: {config.validation_prompt}"
    )

    pips = pips.to(config.device)
    pips.set_progress_bar_config(disable=True)
    autocast_ctx = (
        torch.autocast(config.device.type) 
        if config.device.type == "cuda" 
        else nullcontext()
    )

    imag_edi = []
    with autocast_ctx:
        for _ in range(config.num_validation_images):
            result = pips(
                config.validation_prompt,
                num_inference_steps=config.num_inference_steps,
                image_guidance_scale=config.image_guidance_scale,
                guidance_scale=config.guidance_scale,
                generator=generator,
            )
            imag_edi.append(result.images[0])
            
    return imag_edi

# Prepare parameters
def parse_args():
    parser = argparse.ArgumentParser(description="Training script for InstructPix2Pix.")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to a pretrained model or model identifier from Hugging Face Hub.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of the pretrained model (e.g., a specific branch, tag, or commit ID).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the pretrained model (e.g., 'fp16').",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Name or path of the training dataset. Can be a Hub dataset or a local folder.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="Specific config of the dataset. Leave empty if only one config exists.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "Path to the training data folder (must include a `metadata.jsonl` file). "
            "See Hugging Face documentation: https://huggingface.co/docs/datasets/image_dataset#imagefolder"
        ),
    )
    parser.add_argument("--original_image_column", type=str, default="input_image", help="Column with original images.")
    parser.add_argument("--edited_image_column", type=str, default="edited_image", help="Column with edited images.")
    parser.add_argument("--edit_prompt_column", type=str, default="edit_prompt", help="Column with edit instructions.")
    parser.add_argument(
        "--val_image_url",
        type=str,
        default=None,
        help="URL of an image to edit for validation/debugging during training.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="Prompt to use for generating validation images.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of validation images to generate per evaluation.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging or fast runs, limit the number of training examples.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="instruct-pix2pix-model",
        help="Directory to save model checkpoints and predictions.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache downloaded models and datasets.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Resolution for input images; all images will be resized to this size.",
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="If set, center-crop images after resizing. Otherwise, random crop.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="Randomly flip training images horizontally.",
    )
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size per device.")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Total number of training epochs.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Override epochs and specify total training steps manually.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Accumulate gradients over multiple steps before updating weights.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory at the cost of speed.",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale learning rate based on total batch size and number of GPUs.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="Type of learning rate scheduler to use.",
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Warmup steps for the scheduler.")
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=None,
        help="Probability of dropping condition inputs (image and prompt) during training. "
             "See Section 3.2.1 of the paper: https://arxiv.org/abs/2211.09800",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Use 8-bit Adam optimizer from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TF32 for faster training on Ampere GPUs. "
             "See: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices",
    )
    parser.add_argument("--use_ema", action="store_true", help="Use EMA (Exponential Moving Average) model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        help="Revision of non-EMA model to load from the Hugging Face Hub.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of workers for the dataloader. 0 means main process.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Beta1 coefficient for Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Beta2 coefficient for Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay for Adam optimizer.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for Adam optimizer.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to Hugging Face Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="Token for pushing to the Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Repository name on Hugging Face Hub.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard log directory (default: output_dir/runs/CURRENT_DATETIME_HOSTNAME).",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Use mixed precision training: 'fp16' or 'bf16'.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="Reporting integration for logs: 'tensorboard', 'wandb', 'comet_ml', or 'all'.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local rank.")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a training checkpoint every N steps.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Maximum number of checkpoints to keep.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help='Path to a checkpoint directory to resume training from, or "latest".',
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Use xformers for memory-efficient attention (if installed).",
    )

    return parser.parse_args()

def toNp(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)

def main():
    args = parse_args()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if torch.distributed.get_rank() == 0: 
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if torch.distributed.get_rank() == 0: 
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, low_cpu_mem_usage=True
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant, low_cpu_mem_usage=True
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant, low_cpu_mem_usage=True
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision, low_cpu_mem_usage=True
    )

    # InstructPix2Pix requires 8-channel input (concatenation of the original and edited images),
    # instead of the default 4 channels. 
    # To support this, we modify the original pretrained UNet by creating a new input convolution layer:
    # - The new conv layer accepts 8 input channels.
    # - The weights for the additional channels are initialized to zero.
    # - The existing pretrained weights are copied over for the original channels.
    # Finally, we replace the original input conv layer with the modified one.
    logger.info("Modifying the UNet to accept 8-channel inputs for InstructPix2Pix.")
    channel_in = 8
    channel_out = unet.conv_in.out_channels
    unet.register_to_config(in_channels=channel_in)

    with torch.no_grad():
        new_conv_in = nn.Conv2d(
            channel_in, channel_out, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :8, :, :].copy_(unet.conv_in.weight)
        unet.conv_in = new_conv_in

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Initialize Exponential Moving Average for UNet if enabled
    if args.use_ema:
        ema_unet = EMAModel(
            parameters=unet.parameters(),
            model_cls=UNet2DConditionModel,
            model_config=unet.config
        )

    # Configure memory-efficient attention if requested
    if args.enable_xformers_memory_efficient_attention:
        try:
            import xformers
            from packaging import version
            # Verify xFormers version compatibility
            xformers_now = version.parse(xformers.__version__)
            
            if xformers_now == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 has known GPU compatibility issues. "
                    "Recommended to upgrade to 0.0.17+. "
                    "See https://huggingface.co/docs/diffusers/main/en/optimization/xformers"
                )
            # Enable memory efficient attention
            if hasattr(unet, 'enable_xformers_memory_efficient_attention'):
                unet.enable_xformers_memory_efficient_attention()
            else:
                logger.warning("Current UNet implementation doesn't support xFormers optimization")
                
        except ImportError as e:
            raise ImportError(
                "xFormers package not found. Install with: "
                "pip install xformers"
            ) from e

    def unwrap_model(model):
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    # Save
    torch.save({
        "model": unet.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "step": global_step,
    }, os.path.join(args.output_dir, "checkpoint.pt"))

    # Load
    checkp = torch.load(os.path.join(args.output_dir, "checkpoint.pt"), map_location=device)
    unet.load_state_dict(checkp["model"])
    optimizer.load_state_dict(checkp["optimizer"])
    lr_scheduler.load_state_dict(checkp["lr_scheduler"])
    global_step = checkp["step"]

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    if args.enable_xformers_memory_efficient_attention:
        unet.set_use_memory_efficient_attention_xformers(True)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        num_pro = dist.get_world_size() if dist.is_initialized() else 1
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * num_pro
        )

    # Initialize the optimizer
    # Configure the optimizer based on training settings
    optimizer_class = None
    
    # Handle 8-bit Adam optimizer if enabled
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            from packaging import version
            # Verify bitsandbytes version compatibility
            bnb_version = version.parse(bnb.__version__)
            if bnb_version < version.parse("0.39.0"):
                logger.warning(
                    f"bitsandbytes version {bnb_version} may have performance issues. "
                    "Recommend upgrading to 0.39.0+ for optimal 8-bit Adam performance."
                )
            optimizer_class = bnb.optim.AdamW8bit
            logger.info("Using 8-bit Adam optimizer (bitsandbytes)")
            
        except ImportError as e:
            raise ImportError(
                "8-bit Adam optimizer requires bitsandbytes package. "
                "Install with: pip install bitsandbytes"
            ) from e
    else:
        optimizer_class = torch.optim.AdamW
        logger.info("Using standard AdamW optimizer")

    # Initialize optimizer with configured parameters
    optimizer = optimizer_class(
        params=unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/main/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    col_name = dataset["train"].column_names

    # 6. Get the column names for input/target.
    # Determine column names for original image, edit prompt, and edited image
    col_name = dataset["train"].column_names
    dataset_columns = MAPPINGS_DATA.get(args.dataset_name, col_name[:3])

    def resolve_column(arg_val, default_val, arg_name):
        if arg_val is not None:
            if arg_val not in col_name:
                raise ValueError(
                    f"--{arg_name}='{arg_val}' is invalid. Choose from: {', '.join(col_name)}"
                )
            return arg_val
        return default_val

    original_image_column = resolve_column(args.original_image_column, dataset_columns[0], "original_image_column")
    edit_prompt_column = resolve_column(args.edit_prompt_column, dataset_columns[1], "edit_prompt_column")
    edited_image_column = resolve_column(args.edited_image_column, dataset_columns[2], "edited_image_column")

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(captions):
        inp = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inp.input_ids

    def preprocess_images(egs):
        original_images = np.concatenate(
            [image for image in egs[original_image_column]]
        )
        edited_images = np.concatenate(
            [image for image in egs[edited_image_column]]
        )
        images = np.stack([original_images, edited_images])
        images = torch.tensor(images)
        return images

    def preprocess_train(egs):
        # Preprocess images.
        preprocessed_images = preprocess_images(egs)
        # Since the original and edited images were concatenated before
        # applying the transformations, we need to separate them and reshape
        # them accordingly.
        original, edited = preprocessed_images
        original = original.reshape(-1, 3, args.resolution, args.resolution)
        edited = edited.reshape(-1, 3, args.resolution, args.resolution)

        # Collate the preprocessed images into the `examples`.
        egs["original_pixel_values"] = original
        egs["edited_pixel_values"] = edited

        # Preprocess the captions.
        captions = list(egs[edit_prompt_column])
        egs["input_ids"] = tokenize_captions(captions)
        return egs

    if torch.distributed.get_rank() == 0: 
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def fn_collating(examples):
        # Batch and format pixel data
        def stack_images(key):
            return torch.stack([ex[key] for ex in examples]).to(memory_format=torch.contiguous_format).float()

        return {
            "original_pixel_values": stack_images("original_pixel_values"),
            "edited_pixel_values": stack_images("edited_pixel_values"),
            "input_ids": torch.stack([ex["input_ids"] for ex in examples]),
        }

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=fn_collating,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_pro = args.lr_warmup_steps * dist.get_world_size()
    if args.max_train_steps is None:
        len_dataloader_after_shar = math.ceil(len(train_dataloader) / num_pro)
        num_up_step_one_epoch = math.ceil(len_dataloader_after_shar / args.gradient_accumulation_steps)
        num_train_steps = (
            args.num_train_epochs * num_up_step_one_epoch * num_pro
        )
    else:
        num_train_steps = args.max_train_steps * num_pro

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_pro,
        num_training_steps=num_train_steps,
    )

    unet.to(device)
    optimizer.to(device)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    if args.use_ema:
        ema_unet.to(device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16" and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()  
        autocast_ctx = torch.cuda.amp.autocast(enabled=True)  
    elif args.mixed_precision == "bf16" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        autocast_ctx = torch.cuda.amp.autocast(enabled='bf16')  
    else:
        scaler = None  # No scaler if mixed precision is not enabled
        autocast_ctx = nullcontext()  # Default context if no mixed precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_up_step_one_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_up_step_one_epoch
        if num_train_steps != args.max_train_steps * num_pro:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_dataloader_after_shar}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_up_step_one_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if dist.get_rank() == 0:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir="logs")
        writer.add_text("Config", str(vars(args)))

    # Train!
    total_batch_size = args.train_batch_size * (torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    during_bar = tqdm(range(args.max_train_steps), desc="Steps")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            di = os.listdir(args.output_dir)
            di = di = [d for d in di if os.path.isdir(os.path.join(args.output_dir, d)) and d.startswith("checkpoint")]
            di = sorted(di, key=lambda x: int(x.split("-")[1]))
            path = di[-1] if len(di) > 0 else None
        
        if args.resume_from_checkpoint != "latest" and not os.path.exists(os.path.join(args.output_dir, path)):
            print(f"Specified checkpoint '{args.resume_from_checkpoint}' not found in '{args.output_dir}'. Starting new training.")
            args.resume_from_checkpoint = None

        else:
            logger.info(f"Resuming from checkpoint {path}")
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_up_step_one_epoch
            resume_step = resume_global_step % (num_up_step_one_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    during_bar = tqdm(range(global_step, args.max_train_steps))
    during_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    during_bar.update(1)
                continue

            latents = vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = text_encoder(batch["input_ids"])[0]
            original_image_embeds = vae.encode(batch["original_pixel_values"].to(weight_dtype)).latent_dist.mode()

            if args.conditioning_dropout_prob is not None:
                random_p = torch.rand(latents.shape[0], device=latents.device, generator=generator)
                prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                prompt_mask = prompt_mask.reshape(latents.shape[0], 1, 1)
                null_conditioning = text_encoder(tokenize_captions([""]).to(latents.device))[0]
                encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

                image_mask_dtype = original_image_embeds.dtype
                image_mask = 1 - ((random_p >= args.conditioning_dropout_prob).to(image_mask_dtype) *
                                (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype))
                image_mask = image_mask.reshape(latents.shape[0], 1, 1, 1)
                original_image_embeds = image_mask * original_image_embeds

            concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            model_pred = unet(concatenated_noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if args.use_ema:
                    ema_unet.step(unet.parameters())

                during_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if torch.distributed.get_rank() == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        torch.save(unet.state_dict(), save_path)
                        logger.info(f"Saved checkpoint to {save_path}")

                if global_step >= args.max_train_steps:
                    break
        
        if dist.get_rank() == 0:
            if (
                (args.val_image_url is not None)
                and (args.validation_prompt is not None)
                and (epoch % args.validation_epochs == 0)
            ):
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())
                # The models need unwrapping because for compatibility in distributed training mode.
                pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=unwrap_model(unet),
                    text_encoder=unwrap_model(text_encoder),
                    vae=unwrap_model(vae),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                validate_log(
                    pipeline,
                    args,
                    generator,
                )

                if args.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())
                del pipeline
                torch.cuda.empty_cache()

    # Create the pipeline using the trained modules and save it.
    dist.barrier()
    if dist.get_rank() == 0:
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=unwrap_model(text_encoder),
            vae=unwrap_model(vae),
            unet=unwrap_model(unet),
            revision=args.revision,
            variant=args.variant,
        )
        pipeline.save_pretrained(args.output_dir)

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        if (args.val_image_url is not None) and (args.validation_prompt is not None):
            validate_log(
                pipeline,
                args, 
                generator,
            )

if __name__ == "__main__":
    main()
