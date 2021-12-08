from os import path

project_root = "/work/aleko/dev/vi_project"
videos = path.join(project_root, "Videos")
video_images = path.join(project_root, "Video-images")
predictions_root = path.join(project_root, "predictions")

def annotation_path(video_id):
    return path.join(video_images, f"Video_{video_id}", "annotations.json")

def video_path(video_id):
    return path.join(video_images, f"Video_{video_id}")

def get_video_images_path(video_id, video_type="ambient"):
    return path.join(video_images, f"Video_{video_id}", video_type)


ambient_path = lambda video_id: get_video_images_path(video_id, "ambient")
intensity_path = lambda video_id: get_video_images_path(video_id, "intensity")
range_path = lambda video_id: get_video_images_path(video_id, "range")