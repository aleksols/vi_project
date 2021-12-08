import torch
import numpy as np
import torchvision.transforms as T
import project_paths
from PIL import Image
import matplotlib.image as plt_image
import os
from pycocotools.coco import COCO


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        np_image = np.array(image)
        np_image.transpose((2, 0, 1))

        return torch.as_tensor(np_image, dtype=torch.int32)


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root=project_paths.video_images, video_ids=[0, 1, 2], channels=[0, 1, 2], transforms=None):
        self.transform = transforms
        self.root = root
        self.video_ids = video_ids
        self.channels = channels

        self.cocos = []
        self.ambient_paths = []
        self.intensity_paths = []
        self.range_paths = []
        self.local_img_ids = []

        for vid_id in video_ids:
            amb_root = project_paths.ambient_path(vid_id)
            int_root = project_paths.intensity_path(vid_id)
            rng_root = project_paths.range_path(vid_id)
            coco = COCO(project_paths.annotation_path(vid_id))
            self.local_img_ids += list(coco.imgs.keys())
            for frame_path in sorted(os.listdir(project_paths.ambient_path(vid_id))):
                self.ambient_paths.append(os.path.join(amb_root, frame_path))
                self.intensity_paths.append(os.path.join(int_root, frame_path))
                self.range_paths.append(os.path.join(rng_root, frame_path))
                self.cocos.append(coco)

    def _get_images(self, idx):
        ambient_frame = plt_image.imread(self.ambient_paths[idx])
        intensity_frame = plt_image.imread(self.intensity_paths[idx])
        range_frame = plt_image.imread(self.range_paths[idx])
        frame = np.stack([
            ambient_frame.sum(axis=2) / 3,
            intensity_frame.sum(axis=2) / 3,
            range_frame.sum(axis=2) / 3,
        ],
            axis=2)
        pil_frame = (frame * 255).astype("uint8")
        channel_filter = self.channels if len(self.channels) > 1 else self.channels[0]
        channel_filtered_frame = pil_frame[:, :, channel_filter]
        pil_image = Image.fromarray(channel_filtered_frame)
        if self.transform:
            return self.transform(pil_image)

        return pil_image
    
    def _get_target(self, idx):
        _, anns = self.get_coco_anns(idx)
        boxes = []
        labels = []
        areas = []
        is_crowds = []
        for ann in anns:
            bbox = ann["bbox"]
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = xmin + bbox[2]
            ymax = ymin + bbox[3]
            boxes.append([xmin, ymin, xmax, ymax])
            areas.append(ann["area"])
            labels.append(ann["category_id"])
            is_crowds.append(ann["iscrowd"])

        target = {}
        target["image_id"] = torch.tensor(idx)
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["area"] = torch.as_tensor(areas, dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(is_crowds, dtype=torch.int64)
        return target

    def __len__(self):
        return len(self.ambient_paths)

    def get_coco_anns(self, idx):
        coco = self.cocos[idx]
        coco_img_id = self.local_img_ids[idx]
        ann_ids = coco.getAnnIds(imgIds=coco_img_id)
        anns = coco.loadAnns(ann_ids)
        return coco, anns

    def __getitem__(self, idx):
        img = self._get_images(idx)
        target = self._get_target(idx)
        return img, target


class SquareCocoImages(CocoDataset):

    def _get_images(self, idx):
        path_index = idx // 8
        sub_image = idx % 8

        sub_image_slice = slice(sub_image * 128, (sub_image + 1) * 128)

        ambient_frame = plt_image.imread(self.ambient_paths[path_index])
        intensity_frame = plt_image.imread(self.intensity_paths[path_index])
        range_frame = plt_image.imread(self.range_paths[path_index])
        frame = np.stack([
            ambient_frame.sum(axis=2) / 3,
            intensity_frame.sum(axis=2) / 3,
            range_frame.sum(axis=2) / 3,
        ],
            axis=2)
        pil_frame = (frame * 255).astype("uint8")
        channel_filter = self.channels if len(self.channels) > 1 else self.channels[0]
        channel_filtered_frame = pil_frame[:, sub_image_slice, channel_filter]
        pil_image = Image.fromarray(channel_filtered_frame)
        if self.transform:
            return self.transform(pil_image)

        return pil_image

    def _get_target(self, idx):
        path_index = idx // 8
        sub_image = idx % 8
        img_start = sub_image * 128
        img_stop = (sub_image + 1) * 128

        def crop_width(bbox):
            x_min = bbox[0] - img_start
            x_max = bbox[0] + bbox[2] - img_start
            if x_min > 128 or x_max < 0:
                return False
            # if not (img_start <= x_min <= img_stop or img_start <= x_max <= img_stop):
            #     return False
            x_min = max(x_min, 0)
            x_max = min(x_max, 128)
            if x_min == x_max:
                return False
            return x_min, x_max

        _, anns = self.get_coco_anns(path_index)
        boxes = []
        labels = []
        areas = []
        is_crowds = []
        for ann in anns:
            bbox = ann["bbox"]
            crop = crop_width(bbox)
            if not crop:
                # box = np.array([]).reshape(-1, 4)
                # boxes.append(box)
                # areas.append(None)
                # labels.append(None)
                # is_crowds.append(False)
                continue
            
            xmin, xmax = crop
            ymin = bbox[1]
            ymax = ymin + bbox[3]
            boxes.append([xmin, ymin, xmax, ymax])
            areas.append((xmax - xmin) * (ymax - ymin))
            labels.append(ann["category_id"])
            is_crowds.append(ann["iscrowd"])

        target = {}
        target["image_id"] = torch.tensor(idx)
        if len(boxes) > 0:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        else: 
            target["boxes"] = torch.as_tensor(np.array([]).reshape(-1, 4), dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["area"] = torch.as_tensor(areas, dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(is_crowds, dtype=torch.int64)
        # print(target)
        return target

    def __len__(self):
        return super().__len__() * 8
    


def get_dataloader(vids=[0, 1, 2], channels=[0, 1, 2], batch_size=4, num_workers=0, train=False):
    dataset = CocoDataset(
        video_ids=vids,
        channels=channels,
        transforms=get_transform(train)
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=lambda batch: list(zip(*batch))
    )
    return data_loader

def get_square_dataloader(vids=[0, 1, 2], channels=[0, 1, 2], batch_size=4, num_workers=0, train=False):
    dataset = SquareCocoImages(
        video_ids=vids,
        channels=channels,
        transforms=get_transform(train)
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=lambda batch: list(zip(*batch))
    )
    return data_loader




if __name__ == "__main__":
    dl = get_dataloader(vids=[18], num_workers=4)
    # for img_batch, target_batch in dl:
    #     for target in target_batch:
    #         print(target["image_id"])
    
    ds = dl.dataset
    coco, anns = ds.get_coco_anns(0)
    print(anns)