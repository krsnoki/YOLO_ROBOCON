import os
from PIL import Image

def concatenate_images(blur_folder, sharp_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the list of image files in the blur folder
    blur_files = os.listdir(blur_folder)
    sharp_files =os.listdir(sharp_folder)

    # Iterate through each blur image file
    for blur_file in blur_files:
        # Check if the file has the .png extension
        if blur_file.lower().endswith('.jpg'):
            # Get the corresponding sharp image file
            sharp_file = os.path.join(sharp_folder, os.path.basename(blur_file))

            # Check if both blur and sharp image files exist
            if os.path.isfile(os.path.join(blur_folder, blur_file)) and os.path.isfile(sharp_file):
                # Open the blur and sharp images
                blur_image = Image.open(os.path.join(blur_folder, blur_file))
                sharp_image = Image.open(sharp_file)

                # Concatenate the images horizontally
                concatenated_image = Image.new('RGB', (blur_image.width + sharp_image.width, blur_image.height))
                concatenated_image.paste(blur_image, (0, 0))
                concatenated_image.paste(sharp_image, (blur_image.width, 0))

                # Save the concatenated image
                output_file = os.path.join(output_folder, blur_file)
                concatenated_image.save(output_file)

                print(f"Concatenated image saved: {output_file}")
            else:
                print(f"Corresponding sharp image not found for: {blur_file}")
        else:
            print(f"Skipping non-PNG file: {blur_file}")

# Example usage
blur_folder = r'F:/adi/skipfinal/datasets/UFO-120/lrd'
white_balls = r''
output_folder = r'../output_frames' # Add 'r' before the path to treat it as a raw string'

concatenate_images(blur_folder, sharp_folder, output_folder)