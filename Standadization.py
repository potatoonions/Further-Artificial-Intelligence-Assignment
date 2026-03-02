from PIL import Image
import numpy as np
from pathlib import Path


def standardize_image(image_input, output_size=(256, 256)):
    """
    Standardizes an image to 256x256 RGB format (8 bits per channel).
    
    Parameters:
    -----------
    image_input : str, Path, or PIL.Image.Image
        Path to the image file or PIL Image object
    output_size : tuple
        Target size (width, height). Default is (256, 256)
    
    Returns:
    --------
    PIL.Image.Image
        Standardized image in RGB format, 256x256 pixels
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
    # Using LANCZOS for high-quality downsampling and upsampling
    image_resized = image.resize(output_size, Image.Resampling.LANCZOS)
    
    # Step 3: Ensure 8 bits per channel (8-bit RGB)
    if image_resized.mode != 'RGB':
        image_resized = image_resized.convert('RGB')
    
    return image_resized


def standardize_and_save(image_input, output_path, output_size=(256, 256)):
    """
    Standardizes an image and saves it to disk.
    
    Parameters:
    -----------
    image_input : str, Path, or PIL.Image.Image
        Path to the image file or PIL Image object
    output_path : str or Path
        Path where the standardized image will be saved
    output_size : tuple
        Target size (width, height). Default is (256, 256)
    
    Returns:
    --------
    str
        Path to the saved standardized image
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
    
    Parameters:
    -----------
    image_directory : str or Path
        Directory containing images to standardize
    output_directory : str or Path
        Directory where standardized images will be saved
    output_size : tuple
        Target size (width, height). Default is (256, 256)
    
    Returns:
    --------
    list
        List of processed image paths
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


# Example usage:
if __name__ == "__main__":
    # Single image standardization
    # image = standardize_image("path/to/galaxy_image.jpg")
    # image.show()
    
    # Save standardized image
    # standardize_and_save("path/to/galaxy_image.jpg", "output/standardized_galaxy.png")
    
    # Batch process directory
    # processed = batch_standardize("input_galaxies/", "output_galaxies/")
    # print(f"Processed {len(processed)} images")
    
    pass
