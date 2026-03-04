from PIL import Image
import numpy as np
import json
import argparse
import sys
import asyncio
from pathlib import Path
from typing import Optional


def standardize_image(image_input, output_size=(256, 256)):
    """
    Standardizes an image to 256x256 RGB format (8 bits per channel).
    
    """
    
    # Load image if path is provided
    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input)
    else:
        image = image_input.copy()
    
    # Step 1: Convert to RGB (handles grayscale, RGBA, CMYK, etc.)
    if image.mode == 'RGBA':
        # If image has transparency, convert on white background
        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
        rgb_image.paste(image, mask=image.split()[3])  # Use alpha channel as mask
        image = rgb_image
    elif image.mode == 'CMYK':
        image = image.convert('RGB')
    elif image.mode == 'L':
        # Grayscale to RGB (expand channels)
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        # Any other mode, convert to RGB
        image = image.convert('RGB')
    
    # Step 2: Resize to 256x256
    image_resized = image.resize(output_size, Image.Resampling.LANCZOS)
    
    # Step 3: Ensure 8 bits per channel (8-bit RGB)
    if image_resized.mode != 'RGB':
        image_resized = image_resized.convert('RGB')
    
    return image_resized


def standardize_and_save(image_input, output_path, output_size=(256, 256)):
    """
    Standardizes an image and saves it to disk.
    
    """
    
    standardized_image = standardize_image(image_input, output_size)
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as PNG or JPG (PNG preserves quality)
    standardized_image.save(output_path, quality=95)
    
    return str(output_path)


def batch_standardize(image_directory, output_directory, output_size=(256, 256)):
    """
    Standardizes all images in a directory.
    
    """
    
    image_directory = Path(image_directory)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # Supported image formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    processed_files = []
    
    for image_file in image_directory.iterdir():
        if image_file.suffix.lower() in image_extensions:
            try:
                output_file = output_directory / f"{image_file.stem}_standardized.png"
                standardize_and_save(image_file, output_file, output_size)
                processed_files.append(str(output_file))
                print(f"✓ Standardized: {image_file.name}")
            except Exception as e:
                print(f"✗ Error processing {image_file.name}: {str(e)}")
    
    return processed_files


def load_validation_results(results_file: str) -> list[dict]:
    """
    Load validation results from a JSON file produced by Validation.py
    
    """
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[Error] Validation results file not found: {results_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"[Error] Invalid JSON in validation results file: {results_file}")
        sys.exit(1)


def filter_galaxy_images(results: list[dict]) -> list[Path]:
    """
    Filter validation results to get only confirmed galaxy images
    
    """
    galaxy_images = []
    for result in results:
        if result.get("is_galaxy") is True and result.get("error") is None:
            galaxy_images.append(Path(result["file"]))
    return galaxy_images


def standardize_and_save_galaxy_batch(
    galaxy_images: list[Path], 
    output_directory: str,
    output_size: tuple = (256, 256)
) -> dict:
    """
    Standardize validated galaxy images and save them for classification
    
    """
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Metadata for classification phase
    metadata = {
        "standardized_images": [],
        "failed_images": [],
        "output_size": output_size,
        "output_directory": str(output_dir)
    }
    
    width = 72
    print("\n" + "=" * width)
    print(f"{'STANDARDIZING GALAXY IMAGES':^{width}}")
    print("=" * width)
    
    for idx, image_path in enumerate(galaxy_images, 1):
        try:
            # Create output filename
            output_filename = f"{image_path.stem}_standardized.png"
            output_path = output_dir / output_filename
            
            # Standardize and save
            saved_path = standardize_and_save(image_path, output_path, output_size)
            
            metadata["standardized_images"].append({
                "original": str(image_path),
                "standardized": str(saved_path),
                "size": output_size,
                "format": "PNG"
            })
            
            print(f"  ✓  [{idx:3d}] {image_path.name:<40} → {output_filename}")
            
        except Exception as e:
            metadata["failed_images"].append({
                "original": str(image_path),
                "error": str(e)
            })
            print(f"  ✗  [{idx:3d}] {image_path.name:<40} ERROR: {str(e)}")
    
    print("=" * width)
    print(f"Successfully Standardized: {len(metadata['standardized_images'])}")
    print(f"Failed: {len(metadata['failed_images'])}")
    print(f"Output Directory: {output_dir}")
    print("=" * width + "\n")
    
    # Save metadata for next phase
    metadata_path = output_dir / "standardization_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}\n")
    
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standardize validated galaxy images for classification"
    )
    parser.add_argument(
        "--validation-results",
        type=str,
        required=True,
        help="Path to JSON file from Validation.py containing validation results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="standardized_galaxies/",
        help="Directory to save standardized images (default: standardized_galaxies/)"
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[256, 256],
        metavar=("WIDTH", "HEIGHT"),
        help="Output image size in pixels (default: 256 256)"
    )
    
    args = parser.parse_args()
    
    # Load validation results
    print(f"Loading validation results from: {args.validation_results}")
    results = load_validation_results(args.validation_results)
    
    # Filter galaxy images
    galaxy_images = filter_galaxy_images(results)
    
    if not galaxy_images:
        print("[Error] No validated galaxy images found in results.")
        sys.exit(1)
    
    print(f"Found {len(galaxy_images)} validated galaxy images.\n")
    
    # Standardize and save
    output_size = tuple(args.size)
    metadata = standardize_and_save_galaxy_batch(
        galaxy_images,
        args.output_dir,
        output_size
    )
    
    print(f"Standardization complete! Ready for classification phase.")


if __name__ == "__main__":
    main()
