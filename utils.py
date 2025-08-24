import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

class Process:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def read_img(self):
        imgs = []
        paths = []
        for filename in os.listdir(self.folder_path):
            img_path = os.path.join(self.folder_path, filename)
            img = Image.open(img_path).convert("RGB")
            img = self.tf(img)
            imgs.append(img)
            paths.append(img_path)
        
        return imgs, paths
    
    def colors(self):
        VOC_COLORMAP = [
            (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
            (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
            (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
            (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
            (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)
        ]
        colors = np.array(VOC_COLORMAP, dtype="uint8")

        return colors
