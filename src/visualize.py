from os import path, listdir
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import get_model
import time
from torchvision.io import encode_png, image
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import os
from PIL import Image
import torch
from project_paths import predictions_root

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

    exp_dir = os.path.join(predictions_root, run_dir)
    os.mkdir(exp_dir)
    for threshold in [0.0, 0.1, 0.2, 0.3, 0.5]:
        output_dir = os.path.join(exp_dir, f"{threshold}_confidence")
        os.mkdir(output_dir)
        for i, image in tqdm(enumerate(images), desc="Saving images", total=len(images)):
            gt_boxes, gt_labels = get_boxes_and_labels(targets[i])
            pred_boxes, pred_labels = get_boxes_and_labels(predictions[i], score_threshold=threshold)
        
            img = image[:1 ,: ,:]
            img = img.repeat(3, 1, 1)
            img = img * 255
            img = img.type(torch.uint8)

            drawn_boxes = draw_bounding_boxes(img, gt_boxes, colors="blue", labels=gt_labels) if len(gt_boxes) > 0 else img
            drawn_predictions = draw_bounding_boxes(drawn_boxes, pred_boxes, colors="red", labels=pred_labels)
            img = F.to_pil_image(drawn_predictions, mode="RGB")
            filename = os.path.join(output_dir, f"{i}.PNG")
            img.save(filename)
    return exp_dir

if __name__ == "__main__":
    from dataset import get_dataloader, get_square_dataloader
    from pprint import pprint
    dl = get_dataloader(vids=[18], batch_size=100, train=False, num_workers=2)
    ds = dl.dataset
    # pprint(ds.ambient_paths)
    # pprint(ds.intensity_paths)
    # pprint(ds.local_img_ids)
    
    model = get_model(None, None)
    model.train(False)
    # images = []
    # targets = []
    # c = 0
    # for batch in iter(dl):
    #     print(c)
    #     img, trgt = batch
    #     images += img
    #     targets += trgt
    #     c += 1
    
    images, targets = next(iter(dl))
    print(len(images))
    # print(images)
    # model.train(True)
    # preds = model(images, targets)
    # from pprint import pprint
    # pprint(preds)
    # # preds = None
    # model.train(False)
    print([images[0].shape])
    preds = model(images[:8])
    save_coco_predictions(images[:8], preds, targets[:8])