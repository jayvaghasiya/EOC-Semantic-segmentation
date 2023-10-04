import os
import pandas as pd
import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn
from pathlib import Path
import math

def apply_seg(image,model):
    pixel_values = feature_extractor_inference(image, return_tensors="pt").pixel_values.to(device)
    # print(pixel_values.shape)
    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits.cpu()
    # print(logits.shape)
    upsampled_logits = nn.functional.interpolate(logits,
                    size=image.shape[:-1], 
                    mode='bilinear',
                    align_corners=False)

    seg = upsampled_logits.argmax(dim=1)[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label, :] = color

    color_seg = color_seg[..., ::-1]
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    return img,color_seg
    

parser = argparse.ArgumentParser(description='Training Code')
parser.add_argument('--checkpoints',
                    type=str,
                        help='path/to/trainedcheckpoint')
parser.add_argument('--source',
                    type=str,
                        help='path to image')
parser.add_argument('--classes',
                    type=str,
                        help='path/to/classes.csv')

args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = pd.read_csv(args.classes)
classes = df['name']
palette = df[['r', 'g', 'b']].values
id2label = classes.to_dict()
label2id = {v: k for k, v in id2label.items()}





model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5", ignore_mismatched_sizes=True,
                                                         num_labels=len(id2label), id2label=id2label, label2id=label2id,
                                                         reshape_last_stage=True)

model.load_state_dict(torch.load(args.checkpoints))
model.eval()
model.to(device)
feature_extractor_inference = SegformerFeatureExtractor(do_random_crop=False, do_pad=False)

if args.source.endswith(".mp4") or args.source.endswith(".avi"):
    cap = cv2.VideoCapture(args.source)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    if os.path.exists("./output_video.mp4"):
        out = cv2.VideoWriter('output_video_1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
    else:
        out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
        
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        frame =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img ,color_seg= apply_seg(frame,model)
        out.write(img)
        cv2.imshow("result",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

else:
    image = cv2.imread(args.source)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img,color_seg= apply_seg(image,model)
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.imwrite("result.jpg",img)
    cv2.imwrite("seg.png",color_seg)
    # fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    # axs[0].imshow(img)
    # axs[1].imshow(color_seg)
    # plt.savefig("./result.jpg")
    # plt.show()
