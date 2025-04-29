import os
import cv2
import numpy as np
from tqdm import tqdm

def add_rain_effect(image, intensity=0.5, direction=(-1, 1)):
    height, width, _ = image.shape
    rain_layer = np.zeros((height, width, 1), dtype=np.uint8)
    num_drops = int(intensity * 1000) 
    drop_length = int(intensity * 20)  
    drop_thickness = max(1, int(intensity * 2))

    for _ in range(num_drops):
        x_start = np.random.randint(0, width)
        y_start = np.random.randint(0, height)
        x_end = x_start + int(direction[0] * drop_length)
        y_end = y_start + int(direction[1] * drop_length)
        cv2.line(rain_layer, (x_start, y_start), (x_end, y_end), (255,), thickness=drop_thickness)

    rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)

    rain_overlay = cv2.addWeighted(image, 1, cv2.merge([rain_layer]*3), intensity, 0)
    return rain_overlay

def process_images(input_folder, output_folder, rain_levels=(0.3, 0.7)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    
    for file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(input_folder, file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Failed to read {file}, skipping...")
            continue

        rain_intensity = np.random.uniform(*rain_levels)
        rain_img = add_rain_effect(img, intensity=rain_intensity)

        output_path = os.path.join(output_folder, file)
        cv2.imwrite(output_path, rain_img)

input_folder = ""  
output_folder = ""  
process_images(input_folder, output_folder)
