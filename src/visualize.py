import fiftyone as fo
from os import path, listdir
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import get_model
import time
from torchvision.io import encode_png
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import os
from PIL import Image
import torch

predictions_root = r"C:\Users\aleks\home\dev\vi_project\predictions"

label_strings = {
    1: "car",
    2: "truck", 
    3:"bus",
    4:"motorcycle",
    5:"bicycle",
    6:"scooter",
    7:"person",
    8:"rider",
}

def get_boxes_and_labels(prediction_dict, score_threshold=None):
    boxes = prediction_dict["boxes"]
    labels = [label_strings[e] for e in prediction_dict["labels"].tolist()]
    if score_threshold is None:
        return boxes, labels
    
    filter_indices = [i for i, s in enumerate(prediction_dict["scores"]) if s > score_threshold]

    # return [boxes[i] for i in filter_indices], [labels[i] for i in filter_indices]
    return boxes[filter_indices], [labels[i] for i in filter_indices]



def save_coco_predictions(images, predictions, targets, score_threshold=0.0):
    run_dir = str(time.time())

    output_dir = os.path.join(predictions_root, run_dir)
    os.mkdir(output_dir)
    for i, image in enumerate(images):
        gt_boxes, gt_labels = get_boxes_and_labels(targets[i])
        pred_boxes, pred_labels = get_boxes_and_labels(predictions[i], score_threshold=score_threshold)
        
        img = image[:1 ,: ,:]
        img = img.repeat(3, 1, 1)
        img = img * 255
        img = img.type(torch.uint8)

        drawn_boxes = draw_bounding_boxes(img, gt_boxes, colors="blue", labels=gt_labels)
        drawn_predictions = draw_bounding_boxes(drawn_boxes, pred_boxes, colors="red", labels=pred_labels)
        img = F.to_pil_image(drawn_predictions, mode="RGB")
        filename = os.path.join(output_dir, f"{i}.PNG")
        img.save(filename)

if __name__ == "__main__":
    from dataset import get_dataloader
    dl = get_dataloader(vids=[18], batch_size=1, train=False)
    model = get_model(None, None)
    model.train(False)
    images, target = next(iter(dl))
    # print(len(images))
    print(len(images))
    print(images[0].shape)
    # print(images)
    preds = model(images)
    # preds = None
    save_coco_predictions(images, preds, target)