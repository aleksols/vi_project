import cv2
from os import path, listdir, mkdir
from tqdm import tqdm

videos = "C:\\Users\\aleks\\home\\dev\\vi_project\\Videos"

video_images = "C:\\Users\\aleks\\home\\dev\\vi_project\\Video-images"

lidar_imates = "C:\\Users\\aleks\\home\\dev\\vi_project\\Videos"

def safe_mkdir(dir_path):
    try:
        mkdir(dir_path)
    except:
        pass

def get_coco_datasets():
    data = {}

    for i, video_dir in enumerate(listdir(videos)):
        videoxxx = path.join(videos, video_dir)
        ambient = next((f for f in listdir(videoxxx) if "ambient" in f), None)
        intensity = next((f for f in listdir(videoxxx) if "intensity" in f), None)
        range_vid = next((f for f in listdir(videoxxx) if "range" in f), None)

        data[i] = {
                "ambient": path.join(videoxxx, ambient),
                "intensity": path.join(videoxxx, intensity),
                "range": path.join(videoxxx, range_vid),
                "annotations": None
                }
        if "ano" in videoxxx.lower():
            data[i]["annotations"] = path.join(videoxxx, "instances_default.json")
    return data

def write_vid_to_img(video_path, full_out_path):
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
        frame_name = ["0"] * 6
        frame_name[3] = count // 100
        frame_name[4] = (count - 100 * frame_name[3]) // 10
        frame_name[5] = count % 10
        frame_name = "".join((str(x) for x in frame_name))
        print(type(image))
        print(dir(image))
        break
        cv2.imwrite(path.join(full_out_path, f"frame_{frame_name}.PNG"), image)     # save frame as PNG file
        success, image = vidcap.read()
        count += 1


def main():
    for key, path_dict in tqdm(get_coco_datasets().items()):
        vid_parent_path = path.join(video_images, f"Video_{key}")
        safe_mkdir(vid_parent_path)
        
        for vid_type in ["ambient", "intensity", "range"]:
            vid_type_dir = path.join(vid_parent_path, vid_type)
            safe_mkdir(vid_type_dir)
            write_vid_to_img(path_dict[vid_type], vid_type_dir)

        annotation_path = path_dict["annotations"]
        if annotation_path is not None:
            with open(annotation_path, "r") as f:
                annotations = f.read()
            print(vid_parent_path)
            with open(path.join(vid_parent_path, "annotations.json"), "w") as f:
                f.write(annotations)
                


if __name__ == "__main__":
    # safe_mkdir(video_images)
    # main()

    write_vid_to_img(videos)
    # # from pprint import pprint
    # pprint(get_coco_datasets())
    # import os
    # write_vid_to_img(r"C:\Users\aleks\Downloads\Video00000_ambient.mp4", os.getcwd() + "/test/")