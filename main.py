import torch
import os
import argparse
import numpy as np
from PIL import Image
from torchvision import models
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
from utils import Process

#学習済みモデルの読み込み
weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

#入出力フォルダの指定&作成
parser = argparse.ArgumentParser()
parser.add_argument("--input_folder_path", type=str, default="images")
parser.add_argument("--output_folder_path", type=str, default="output")
parser.add_argument("--mask_folder_path", type=str, default="mask")
args = parser.parse_args()

input_folder_path = args.input_folder_path
output_folder_path = args.output_folder_path
mask_folder_path = args.mask_folder_path

if not os.path.exists(input_folder_path):
    print("フォルダがありません")

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

if not os.path.exists(mask_folder_path):
    os.makedirs(mask_folder_path)

#class Processの読み込み
process = Process(input_folder_path)
imgs, paths = process.read_img() #画像のパス、配列取得
colors = process.colors() #カラーパレットの取得

#実行
count = 0
for i, j in zip(imgs, paths):
    count += 1
    
    #推論
    with torch.no_grad():
        output = model(i.unsqueeze(0))["out"] #[N,C,h,w]
    pred = output[0].argmax(0).cpu().numpy().astype(np.uint8)
    
    #マスクの生成
    mask = Image.fromarray(pred)
    mask.putpalette(colors)
    mask = mask.convert("RGBA")

    #マスクの保存
    maskname = f"mask_{count}.png"
    mask_path = os.path.join(mask_folder_path, maskname)
    mask.save(mask_path)

    #元画像の読み込み
    ori = Image.open(j).convert("RGBA")

    #マスク画像の生成
    masked_img = Image.blend(ori, mask, alpha=0.5)

    #マスク画像の保存
    filename = f"masked_img_{count}.png"
    masked_img_path = os.path.join(output_folder_path, filename)
    masked_img.save(masked_img_path)
    print(f"{filename} saved.")
