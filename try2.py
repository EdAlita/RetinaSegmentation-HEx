
import os
import numpy as np
from skimage import io, color, filters
import csv

# Define the parameters for the Gabor filter
freqs = [0.05, 0.1, 0.15]
theta = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# Load the image
img_path = "/Users/taiaburrahman/Desktop/git/RetinaSegmentation-HEx/Results/HardExodus/Tests/IDRiD_56.jpg"
img = io.imread(img_path)

# # Convert the image to grayscale
# gray_img = color.rgb2gray(img)

# Extract the Gabor features
gabor_features = []
for freq in freqs:
    for angle in theta:
        filtered_img_real, filtered_img_imag = filters.gabor(img, frequency=freq, theta=angle)
        gabor_features.append(filtered_img_real.mean())
        gabor_features.append(filtered_img_imag.mean())

# Save the features to a CSV file
csv_path = "/Users/taiaburrahman/Desktop/git/RetinaSegmentation-HEx/gabor.csv"

# If the CSV file does not exist, create it and write the header row
if not os.path.exists(csv_path):
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['image_name', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10', 'feature_11', 'feature_12'])
    
# Write the image name and Gabor features to the CSV file
with open(csv_path, mode='a', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow([os.path.basename(img_path)] + gabor_features)

