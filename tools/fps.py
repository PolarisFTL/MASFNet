import torch
import torchvision.models as models
import time
from nets.yolo import MSFNet_Head

model_state_dict = torch.load('')
input_shape = [416, 416]
anchors_mask = [[3, 4, 5], [1, 2, 3]]
num_classes = 5
phi = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MSFNet_Head(anchors_mask, num_classes, phi=phi).to(device)
model.load_state_dict(model_state_dict)
input_data = torch.randn(1, 3, 416, 416).cuda()  
model = model.cuda()

with torch.no_grad():
    model.eval()
    total_time = 0
    num_iterations = 2000
    for _ in range(num_iterations):
        start_time = time.time()
        output = model(input_data)
        end_time = time.time()
        total_time += end_time - start_time
    average_fps = num_iterations / total_time
    print("Average FPS: {:.2f}".format(average_fps))
