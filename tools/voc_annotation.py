import os
import random
import xml.etree.ElementTree as ET
import numpy as np
from utils.utils import get_classes

annotation_mode     = 2
VOCdevkit_path      = ''
VOCdevkit_sets      = [('2007', 'train'), ('2007', 'test')]
data_name           = 'rtts'  
split_output_dir    = 'split_dataset'  
classes_path        = f'model_data/{data_name}_classes.txt'
os.makedirs(split_output_dir, exist_ok=True)  

classes, _ = get_classes(classes_path)

photo_nums = np.zeros(len(VOCdevkit_sets))
nums       = np.zeros(len(classes))

def convert_annotation(year, image_id, list_file):
    in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml'%(year, image_id)), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult') is not None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)),
             int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)),
             int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        nums[cls_id] += 1

if __name__ == "__main__":
    random.seed(0)
    print("Generate train/val txt with bbox annotations.")
    type_index = 0
    for year, image_set in VOCdevkit_sets:
        image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt'%(year, image_set)), encoding='utf-8').read().strip().split()

        output_file_name = os.path.join(split_output_dir, f'{image_set}_{data_name}.txt')
        list_file = open(output_file_name, 'w', encoding='utf-8')
        
        for image_id in image_ids:
            list_file.write('%s/VOC%s/JPEGImages/%s.jpg'%(os.path.abspath(VOCdevkit_path), year, image_id))
            convert_annotation(year, image_id, list_file)
            list_file.write('\n')
        photo_nums[type_index] = len(image_ids)
        type_index += 1
        list_file.close()
    print(f"Finished writing split_dataset/train_{data_name}.txt and val_{data_name}.txt")

    def printTable(List1, List2):
        for i in range(len(List1[0])):
            print("|", end=' ')
            for j in range(len(List1)):
                print(List1[j][i].rjust(int(List2[j])), end=' ')
                print("|", end=' ')
            print()

    str_nums = [str(int(x)) for x in nums]
    tableData = [classes, str_nums]
    colWidths = [max(len(item) for item in col) for col in tableData]
    printTable(tableData, colWidths)
