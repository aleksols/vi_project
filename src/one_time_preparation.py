import os
import sys
import project_paths

root = os.path.join(project_paths.project_root, "Videos")

for video_name in sorted(os.listdir(root)):
    # print(video_name)

    working_dir = os.path.join(root, video_name)
    print(working_dir)
    try:
        coco_dir = next(filter(lambda x: "coco" in x and "zip" not in x, os.listdir(working_dir)))
    except:
        continue
    ann_path = os.path.join(working_dir, coco_dir, "annotations", "instances_default.json")
    target_path = os.path.join(working_dir, "annotations.json")
    os.system(f"mv {ann_path} {target_path}")