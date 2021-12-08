import project_paths
import os
from vid_to_img import write_rgb_vid_to_img, safe_mkdir

def combine_images():
    video_dirs = sorted(os.listdir(project_paths.videos))
    print(video_dirs)
    for video in video_dirs:
        full_path = os.path.join(project_paths.videos, video)
        out_path = os.path.join(project_paths.project_root, "yolo_images", "images", video.split("_")[0])
        safe_mkdir(out_path)
        write_rgb_vid_to_img(full_path, out_path)
    # frame_names = sorted(os.listdir(os.path.join))

def unzip_yolo():
    lidar_root = os.path.join(project_paths.project_root, "LiDAR-videos")
    for vid in os.listdir(lidar_root):
        vid_path = os.path.join(lidar_root, vid)
        yolo_zip = next(filter(lambda x: "yolo.zip" in x ,os.listdir(vid_path)), None)
        if yolo_zip is None:
            continue
        yolo_zip = os.path.join(vid_path, yolo_zip)
        os.system(f"unzip {yolo_zip} -d {vid_path}")

def copy_labels():
    lidar_root = os.path.join(project_paths.project_root, "LiDAR-videos")
    for vid in os.listdir(lidar_root):
        obj_data = os.path.join(lidar_root, vid, "obj_train_data")
        # print(sorted(os.listdir(obj_data)))
        target_dir = os.path.join(project_paths.project_root, "yolo_images", "labels", vid.split("_")[0])
        safe_mkdir(target_dir)
        os.system(f"cp {obj_data}/* {target_dir}")
        # print(target_dir)

if __name__ == "__main__":
    copy_labels()
