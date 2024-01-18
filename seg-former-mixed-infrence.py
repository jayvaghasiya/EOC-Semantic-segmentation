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
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (
    unary_from_labels,
    create_pairwise_bilateral,
    create_pairwise_gaussian,
)
from PIL import Image


def crf(original_image, annotated_image, output_image):
    annotated_image = annotated_image.astype(np.uint32)
    annotated_label = (
        annotated_image[:, :, 0].astype(np.uint32)
        + (annotated_image[:, :, 1] << 8).astype(np.uint32)
        + (annotated_image[:, :, 2] << 16).astype(np.uint32)
    )

    colors, labels = np.unique(annotated_label, return_inverse=True)

    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = colors & 0x0000FF
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    n_labels = len(set(labels.flat))

    print("No of labels in the Image are ")
    print(n_labels)

    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

    U = unary_from_labels(labels, n_labels, gt_prob=0.75, zero_unsure=False)
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(
        sxy=(3, 3),
        compat=3,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    d.addPairwiseBilateral(
        sxy=(80, 80),
        srgb=(13, 13, 13),
        rgbim=original_image,
        compat=10,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    Q = d.inference(5)

    MAP = np.argmax(Q, axis=0)

    MAP = colorize[MAP, :]
    cv2.imwrite(output_image, MAP.reshape(original_image.shape))
    MAP = MAP.reshape(original_image.shape)

    gray_seg = cv2.cvtColor(MAP, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(gray_seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    approx_contours = [
        cv2.approxPolyDP(contour, 0.004 * cv2.arcLength(contour, True), True)
        for contour in contours
    ]

    smooth_mask = np.zeros_like(gray_seg)
    cv2.drawContours(smooth_mask, approx_contours, -1, (255), thickness=cv2.FILLED)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    smooth_mask = cv2.morphologyEx(MAP, cv2.MORPH_CLOSE, kernel, iterations=2)

    return smooth_mask


def apply_seg(image, dense_model,lane_model):
    pixel_values = feature_extractor_inference(
        image, return_tensors="pt"
    ).pixel_values.to(device)
    print(pixel_values.shape)
    dense_outputs = dense_model(pixel_values=pixel_values)
    lane_outputs = lane_model(pixel_values=pixel_values)

    dense_logits = dense_outputs.logits.cpu()
    lane_logits = lane_outputs.logits.cpu()

    # print(logits.shape)
    dense_upsampled_logits = nn.functional.interpolate(
        dense_logits, size=image.shape[:-1], mode="bilinear", align_corners=False
    )
    lane_upsampled_logits = nn.functional.interpolate(
        lane_logits, size=image.shape[:-1], mode="bilinear", align_corners=False
    )

    dense_seg = dense_upsampled_logits.argmax(dim=1)[0]
    lane_seg = lane_upsampled_logits.argmax(dim=1)[0]


    # print("****************************************************",upsampled_logits.argmax(dim=1).shape)
    dense_color_seg = np.zeros((dense_seg.shape[0], dense_seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(dense_palette):
        dense_color_seg[dense_seg == label, :] = color
    
    lane_color_seg = np.zeros((lane_seg.shape[0], lane_seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(lane_palette):
        lane_color_seg[dense_seg == label, :] = color

    dense_color_seg = dense_color_seg[..., ::-1]
    lane_color_seg = lane_color_seg[..., ::-1]
    lane_mask = np.any(lane_color_seg != [0, 0, 0], axis=-1)
    merged_mask = np.copy(dense_color_seg)
    merged_mask[lane_mask] = lane_color_seg[lane_mask]
    merged_mask = np.clip(merged_mask, 0, 255).astype(np.uint8)
    color_seg = crf(image, merged_mask, "sample.jpg")
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    return img, color_seg


parser = argparse.ArgumentParser(description="Training Code")
parser.add_argument("--dense-checkpoints", type=str, help="path/to/trainedcheckpoint")
parser.add_argument("--lane-checkpoints", type=str, help="path/to/trainedcheckpoint")
parser.add_argument("--source", type=str, help="path to image")
parser.add_argument("--dense-classes", type=str, help="path/to/classes.csv")
parser.add_argument("--lane-classes", type=str, help="path/to/classes.csv")


args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df_dense= pd.read_csv(args.dense_classes)
dense_classes = df_dense["name"]
dense_palette = df_dense[["r", "g", "b"]].values
dense_id2label = dense_classes.to_dict()
dense_label2id = {v: k for k, v in dense_id2label.items()}
df_lane= pd.read_csv(args.lane_classes)
lane_classes = df_lane["name"]
lane_palette = df_lane[["r", "g", "b"]].values
lane_id2label = lane_classes.to_dict()
lane_label2id = {v: k for k, v in lane_id2label.items()}

dense_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b5",
    ignore_mismatched_sizes=True,
    num_labels=len(dense_id2label),
    id2label=dense_id2label,
    label2id=dense_label2id,
    reshape_last_stage=True,
)

dense_model.load_state_dict(torch.load(args.dense_checkpoints))
dense_model.eval()
dense_model.to(device)

lane_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b5",
    ignore_mismatched_sizes=True,
    num_labels=len(lane_id2label),
    id2label=lane_id2label,
    label2id=lane_label2id,
    reshape_last_stage=True,
)

lane_model.load_state_dict(torch.load(args.lane_checkpoints))
lane_model.eval()
lane_model.to(device)

feature_extractor_inference = SegformerFeatureExtractor(
    do_random_crop=False, do_pad=False
)

if args.source.endswith(".mp4") or args.source.endswith(".avi"):
    cap = cv2.VideoCapture(args.source)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    if os.path.exists("./output_video.mp4"):
        out = cv2.VideoWriter(
            "output_video_1.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (frame_width, frame_height),
        )
    else:
        out = cv2.VideoWriter(
            "output_video.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (frame_width, frame_height),
        )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img, color_seg = apply_seg(frame, dense_model,lane_model)
        out.write(img)
        cv2.imshow("result", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


else:
    files = os.listdir(args.source)
    for i in files:
        image = cv2.imread(f"{args.source}/{i}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img, color_seg = apply_seg(image, dense_model,lane_model)
        # cv2.imshow("img",img)
        # cv2.imshow("seg",color_seg)
        cv2.imwrite(f'./final-output-mixed/{i.replace(".jpg",".png")}', color_seg)
        cv2.imwrite(f'./final-output-mixed/{i.replace(".jpg","")}-mixed.jpg', img)
        # cv2.waitKey(0)
    #     os.makedirs((args.source).replace("Doyle-Town","output-13-10/Doyle-Town-output-dense"),exist_ok=True)
    #     os.makedirs((args.source).replace("Doyle-Town","output-13-10/Doyle-Town-output-seg-dense"),exist_ok=True)
    #     os.makedirs((args.source).replace("Doyle-Town","output-13-10/Doyle-Town-output-lane"),exist_ok=True)
    #     os.makedirs((args.source).replace("Doyle-Town","output-13-10/Doyle-Town-output-seg-lane"),exist_ok=True)
    #     if args.classes == 'lane-classes.csv':
    #         cv2.imwrite(f'{(args.source).replace("Doyle-Town","output-13-10/Doyle-Town-output-lane")}/{i}',img)
    #         cv2.imwrite(f'{(args.source).replace("Doyle-Town","output-13-10/Doyle-Town-output-seg-lane")}/{i}',color_seg)
    #     else:
    #         cv2.imwrite(f'{(args.source).replace("Doyle-Town","output-13-10/Doyle-Town-output-dense")}/{i}',img)
    #         cv2.imwrite(f'{(args.source).replace("Doyle-Town","output-13-10/Doyle-Town-output-seg-dense")}/{i}',color_seg)
    # # fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    # axs[0].imshow(img)
    # axs[1].imshow(color_seg)
    # plt.savefig("./result.jpg")
    # plt.show()
