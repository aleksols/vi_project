import cv2
import os
import project_paths
import numpy as np

def sort_values(key):
    return int(key.split(".")[0])

def img_to_vid(vid_name, *img_path):
    image_folder = os.path.join(project_paths.project_root, *img_path)
    video_name = os.path.join(project_paths.project_root, "video_outputs", vid_name)

    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".PNG")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 6, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


def combine_and_convert(vid_name, *img_path):
    image_folder = os.path.join(project_paths.project_root, *img_path)
    video_name = os.path.join(project_paths.project_root, "video_outputs", vid_name)


    images = [img for img in sorted(os.listdir(image_folder), key=sort_values) if img.endswith(".PNG")]
    if len(images) % 8 != 0:
        print("incorrect number of images", len(images))
        return
    # print(images[0:8])
    image_batches = [images[i:i+8] for i in range(0, len(images), 8)]
    # print(len(image_batches))
    
    # print(images[0][0])
    frame = cv2.imread(os.path.join(image_folder, image_batches[0][0]))
    
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 6, (width * 8,height))

    for batch in image_batches:
        # output_frame = np.array([]).reshape(128, 128, 3)
        # output_frame = np.empty_like(frame)
        output_frames = []
        for image in batch:
            # print("hello")
            print(image)
            output_frames.append(cv2.imread(os.path.join(image_folder, image)))

        output_frame = np.concatenate(*[output_frames], axis=1)
        # print(output_frame.shape)
        #     # print((output_frame == frame).all())
        # exit()
        video.write(output_frame)

    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    # img_to_vid("video3.avi", "yolov5", "runs", "detect", "exp2")
    img_to_vid("resnet_vid_latest.avi", "predictions/1638968861.471537/0.3_confidence")