from PIL import Image
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from yolo import YOLO


if __name__ == "__main__":
    mode       = '' # predict or dir_predict
    model_path = '' # the path of the model
    data_name  = ''  
    dir_origin_path = ''
    dir_save_path   = ''
    classes_path = f'model_data/{data_name}_classes.txt' 
  
    yolo = YOLO(model_path=model_path, classes_path=classes_path)

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop = False, count=True)
                r_image.show()

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'dir_predict'.")
