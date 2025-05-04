import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from diffusers import StableDiffusionInstructPix2PixPipeline
from transformers import CLIPTokenizer
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory for saving results
output_dir = "./inference_results_2"
os.makedirs(output_dir, exist_ok=True)

# Load the fine-tuned model from HuggingFace
model_id = "Stevenzhanshi/instruct-pix2pix-model"
print(f"Loading model from {model_id}...")
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    safety_checker=None,
)
pipe = pipe.to(device)

# Load the dataset from HuggingFace
dataset_id = "Aurora1609/RoboTwin"
print(f"Loading dataset from {dataset_id}...")
dataset = load_dataset(dataset_id)
print(f"Dataset keys: {dataset['train'].features}")


def tensor_to_pil(tensor_data):
    """
    Convert tensor data from the dataset to a PIL Image.
    Handles various input formats and shapes.
    """
    # Convert to numpy array if not already
    if not isinstance(tensor_data, np.ndarray):
        tensor_data = np.array(tensor_data)

    # Check and fix dimensions
    if len(tensor_data.shape) == 4 and tensor_data.shape[0] == 1:
        # Remove batch dimension if present
        tensor_data = tensor_data[0]

    # Debug information
    print(
        f"Tensor shape: {tensor_data.shape}, dtype: {tensor_data.dtype}, min: {tensor_data.min()}, max: {tensor_data.max()}"
    )

    # Ensure we have a valid image format (H, W, 3) for RGB
    if len(tensor_data.shape) == 3 and tensor_data.shape[2] == 3:
        # Scale to 0-255 range if in 0-1
        if tensor_data.max() <= 1.0:
            tensor_data = (tensor_data * 255).astype(np.uint8)
        return Image.fromarray(tensor_data)
    else:
        raise ValueError(
            f"Unexpected tensor shape: {tensor_data.shape}. Expected (H, W, 3) for RGB image."
        )


# Function to run inference and save results
def run_inference(dataset_split="train", num_samples=10):
    results = []

    # Create a list of sample indices to process
    sample_indices = list(range(min(num_samples, len(dataset[dataset_split]))))

    print(f"Running inference on {len(sample_indices)} samples...")

    for i in tqdm(sample_indices):
        # Get sample directly without using DataLoader
        sample = dataset[dataset_split][i]

        try:
            # Get the before image and prompt
            before_img = sample["before"]
            prompt = sample["prompt"]
            actual_after = sample["after"]

            # Print shapes for debugging
            print(f"Sample {i}:")
            print(f"  Before shape: {np.array(before_img).shape}")
            print(f"  After shape: {np.array(actual_after).shape}")
            print(f"  Prompt: {prompt}")

            # Convert to PIL Image for the pipeline
            before_pil = tensor_to_pil(before_img)
            actual_after_pil = tensor_to_pil(actual_after)

            # Run inference
            with torch.no_grad():
                image = pipe(
                    prompt,
                    image=before_pil,
                    num_inference_steps=20,
                    image_guidance_scale=1.5,
                    guidance_scale=7.0,
                ).images[0]

            # Save results
            before_pil.save(os.path.join(output_dir, f"sample_{i}_before.png"))
            image.save(os.path.join(output_dir, f"sample_{i}_predicted.png"))
            actual_after_pil.save(os.path.join(output_dir, f"sample_{i}_actual.png"))

            # Create a figure with the three images side by side
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(np.array(before_pil))
            axes[0].set_title("Before")
            axes[0].axis("off")

            axes[1].imshow(np.array(image))
            axes[1].set_title(f"Predicted (Prompt: {prompt})")
            axes[1].axis("off")

            axes[2].imshow(np.array(actual_after_pil))
            axes[2].set_title("Actual After")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"comparison_{i}.png"))
            plt.close()

            results.append(
                {
                    "before": before_pil,
                    "predicted": image,
                    "actual_after": actual_after_pil,
                    "prompt": prompt,
                }
            )

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            import traceback

            traceback.print_exc()

    return results


# Main execution
if __name__ == "__main__":
    # First, examine a single sample to understand the data structure
    try:
        sample = dataset["train"][0]
        print("\nExamining first sample structure:")
        print(f"Before data: {type(sample['before'])}")
        print(f"Shape of before data array: {np.array(sample['before']).shape}")
        print(f"Prompt: {sample['prompt']}")

        # Try to convert the first image to understand the structure better
        print("\nAttempting to convert first image...")

        # If the structure is deeply nested, we need to extract the actual image data
        before_data = np.array(sample["before"])

        # Check if we need to reshape or extract the image data differently
        if len(before_data.shape) >= 3:
            print(f"Image shape: {before_data.shape}")

            # If this is a valid image shape (H,W,3), we can try to convert it
            try:
                if before_data.shape[-1] == 3:  # Check if the last dimension is 3 (RGB)
                    # Scale to 0-255 if needed
                    if before_data.max() <= 1.0:
                        before_data = (before_data * 255).astype(np.uint8)
                    img = Image.fromarray(before_data)
                    img.save(os.path.join(output_dir, "sample_first_image.png"))
                    print("Successfully saved first image!")
            except Exception as e:
                print(f"Failed basic conversion: {e}")

                # Try alternative formats/reshaping if needed
                try:
                    # If the data is in a format like [[[r,g,b],[r,g,b],...]], we may need to reshape
                    reshaped = before_data.reshape(
                        before_data.shape[0], before_data.shape[1], 3
                    )
                    reshaped = (reshaped * 255).astype(np.uint8)
                    img = Image.fromarray(reshaped)
                    img.save(os.path.join(output_dir, "sample_reshaped_image.png"))
                    print("Successfully saved reshaped image!")
                except Exception as e2:
                    print(f"Failed reshape attempt: {e2}")

    except Exception as e:
        print(f"Error examining sample: {e}")

    # Set how many samples you want to run inference on
    num_inference_samples = 50

    try:
        results = run_inference(
            dataset_split="train", num_samples=num_inference_samples
        )
        print(f"Inference completed. Results saved in {output_dir}/")
    except Exception as e:
        print(f"Inference failed: {e}")
