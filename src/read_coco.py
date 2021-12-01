from pycocotools.coco import COCO
import numpy as np
import os
from project_paths import *

# annFile = os.path.normpath(r"C:\Users\aleks\home\dev\vi_project\Videos\Video00000_Ano\instances_default.json")
annFile = annotation_path(video_id=0)


# initialize COCO api for instance annotations
coco=COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

# nms = set([cat['supercategory'] for cat in cats])
# print('COCO supercategories: \n{}'.format(' '.join(nms)))

catIds = coco.getCatIds(catNms=['car','truck','bus']);
print(catIds)
annIds = coco.getAnnIds(catIds=catIds );
print(annIds)
anns = coco.loadAnns(annIds)
imgIds = set((ann["image_id"] for ann in anns))
imgIds = coco.getImgIds(imgIds=imgIds)
print(imgIds)
# img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
img = coco.loadImgs(imgIds)[0]
print(img)
print(img["file_name"])
# I = io.imread(img['coco_url'])
# plt.axis('off')
# plt.imshow(I)
# plt.show()