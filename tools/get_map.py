import os
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_map
from yolo import YOLO

class MAPCalculator:
    def __init__(self, 
                 model_path, 
                 classes_path, 
                 vocdevkit_path, 
                 map_out_path, 
                 data_name='', 
                 confidence=0.001, 
                 nms_iou=0.5,
                 min_overlap=0.5,
                 score_threshold=0.5,
                 map_vis=False):
        
        self.model_path = model_path
        self.classes_path = classes_path
        self.vocdevkit_path = vocdevkit_path
        self.map_out_path = map_out_path
        self.data_name = data_name
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.min_overlap = min_overlap
        self.score_threshold = score_threshold
        self.map_vis = map_vis

        self.class_names, _ = get_classes(self.classes_path)
        self.image_ids = self.load_image_ids()
        self.yolo = None

        self.prepare_directories()

    def load_image_ids(self):
        txt_path = os.path.join(self.vocdevkit_path, "VOC2007/ImageSets/Main/test.txt")
        return open(txt_path).read().strip().split()

    def prepare_directories(self):
        os.makedirs(self.map_out_path, exist_ok=True)
        os.makedirs(os.path.join(self.map_out_path, 'ground-truth'), exist_ok=True)
        os.makedirs(os.path.join(self.map_out_path, 'detection-results'), exist_ok=True)
        os.makedirs(os.path.join(self.map_out_path, 'images-optional'), exist_ok=True)

    def load_model(self):
        print("Loading model...")
        self.yolo = YOLO(
            model_path=self.model_path,
            classes_path=self.classes_path,
            confidence=self.confidence,
            nms_iou=self.nms_iou
        )
        print("Model loaded.")

    def generate_predictions(self):
        if self.yolo is None:
            self.load_model()

        print("Generating prediction results...")
        for image_id in tqdm(self.image_ids):
            if self.data_name in ['rain', 'snow']:
                image_path = os.path.join(self.vocdevkit_path, f"VOC2007/{self.data_name}", image_id + ".jpg")
            else:
                image_path = os.path.join(self.vocdevkit_path, "VOC2007/JPEGImages", image_id + ".jpg")
            
            image = Image.open(image_path)
            if self.map_vis:
                image.save(os.path.join(self.map_out_path, "images-optional", image_id + ".jpg"))
            
            self.yolo.get_map_txt(image_id, image, self.class_names, self.map_out_path)
        print("Prediction results generated.")

    def generate_ground_truth(self):
        print("Generating ground truth files...")
        for image_id in tqdm(self.image_ids):
            annotation_path = os.path.join(self.vocdevkit_path, "VOC2007/Annotations", image_id + ".xml")
            root = ET.parse(annotation_path).getroot()
            with open(os.path.join(self.map_out_path, "ground-truth", image_id + ".txt"), "w") as f:
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult') is not None and int(obj.find('difficult').text) == 1:
                        difficult_flag = True

                    obj_name = obj.find('name').text
                    if obj_name not in self.class_names:
                        continue

                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text

                    if difficult_flag:
                        f.write(f"{obj_name} {left} {top} {right} {bottom} difficult\n")
                    else:
                        f.write(f"{obj_name} {left} {top} {right} {bottom}\n")
        print("Ground truth files generated.")

    def calculate_map(self):
        print("Calculating mAP...")
        get_map(
            MINOVERLAP=self.min_overlap,
            draw_plot=True,
            score_threhold=self.score_threshold,
            path=self.map_out_path
        )
        print("mAP calculation complete.")

    def run(self, map_mode=0):
        if map_mode in [0, 1]:
            self.generate_predictions()
        if map_mode in [0, 2]:
            self.generate_ground_truth()
        if map_mode in [0, 3]:
            self.calculate_map()

if __name__ == "__main__":
    data_name = ''
    vocdevkit_path = '' 
    model_path = ''
    classes_path = f'model_data/{data_name}_classes.txt' 
    map_out_path = f'map_out-{data_name}'
    
    calculator = MAPCalculator(
        model_path=model_path,
        classes_path=classes_path,
        vocdevkit_path=vocdevkit_path,
        map_out_path=map_out_path,
        data_name=data_name,
        confidence=0.001,
        nms_iou=0.5,
        min_overlap=0.5,
        score_threshold=0.5,
        map_vis=False
    )
    
    calculator.run(map_mode=0)  
