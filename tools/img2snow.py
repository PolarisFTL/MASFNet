import cv2
import numpy as np
import os

def generate_snow_layer(image_size, intensity=0.5):
    h, w = image_size
    snow_layer = np.zeros((h, w), dtype=np.float32)

    num_snowflakes = int(h * w * intensity * 0.02)
    for _ in range(num_snowflakes):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        radius = np.random.randint(1, 5) 
        intensity = np.random.uniform(0.6, 1.0) 
        cv2.circle(snow_layer, (x, y), radius, intensity, -1)
    
    snow_layer = cv2.GaussianBlur(snow_layer, (7, 7), 0)
    snow_layer = np.clip(snow_layer, 0, 1)  

def add_snow_effect_physical(image, intensity=0.5, light=0.8):

    h, w, c = image.shape
    snow_layer = generate_snow_layer((h, w), intensity)
    
    A = np.array([255, 255, 255], dtype=np.float32) / 255.0
    A = A * light
    
    t = 1 - snow_layer[..., np.newaxis]
    
    image_normalized = image.astype(np.float32) / 255.0
    output = image_normalized * t + A * (1 - t)
    output = np.clip(output, 0, 1) * 255
    return output.astype(np.uint8)

def process_dataset(input_folder, output_folder, intensity=0.5, light=0.8):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        if file_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
            image = cv2.imread(input_path)
            if image is None:
                continue
            
            snow_image = add_snow_effect_physical(image, intensity, light)
            cv2.imwrite(output_path, snow_image)
            print(f"Processed: {file_name}")

input_folder = ""
output_folder = ""
process_dataset(input_folder, output_folder, intensity=0.7, light=0.9)
