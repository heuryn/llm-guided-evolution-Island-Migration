import os
from PIL import Image
import glob
import re

# Define the directory containing the PNG files
input_dir = 'data/pareto_fronts'
output_file = 'data/pareto_fronts/output.gif'
output_file2 = 'data/pareto_fronts/output2.gif'
pattern = os.path.join(input_dir, 'pareto_gen_*.png')
pattern2 = os.path.join(input_dir, 'pareto_norm_grid_gen_*.png')

# Print all files in the directory for debugging
try:
    all_files = os.listdir(input_dir)
    print(f"All files in the directory '{input_dir}':")
    for filename in all_files:
        print(filename)
except FileNotFoundError:
    print(f"The directory {input_dir} does not exist.")
    raise

# Gather the list of image files
image_files = glob.glob(pattern)
image_files2 = glob.glob(pattern2)

# Print the pattern and the list of files found for debugging
print(f"Looking for files matching pattern: {pattern}")
print(f"Files found: {image_files}")

print(f"Looking for files matching pattern: {pattern2}")
print(f"Files found: {image_files2}")

# Custom sort function to sort files numerically based on the number in the filename
def numerical_sort(value):
    parts = re.findall(r'\d+', value)
    return int(parts[-1]) if parts else float('inf')

# Sort the files numerically
image_files = sorted(image_files, key=numerical_sort)
image_files2 = sorted(image_files2, key=numerical_sort)

# Ensure there are images to be processed
if not image_files:
    raise ValueError(f"No images found in {input_dir} matching pattern {pattern}")
if not image_files2:
    raise ValueError(f"No images found in {input_dir} matching pattern {pattern2}")

# Load all images into a list
images = [Image.open(image_file) for image_file in image_files]
images2 = [Image.open(image_files) for image_files in image_files2]

# Save the images as a GIF
# with image frames lasting 200 milliseconds
images[0].save(output_file, save_all=True, append_images=images[1:], duration=500, loop=0)
images2[0].save(output_file2, save_all=True, append_images=images2[1:], duration=500, loop=0)

print(f"GIF created successfully: {output_file}")
print(f"GIF created successfully: {output_file2}")