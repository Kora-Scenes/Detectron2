# !python -m pip install pyyaml==5.1
# # Detectron2 has not released pre-built binaries for the latest pytorch (https://github.com/facebookresearch/detectron2/issues/4053)
# # so we install from source instead. This takes a few minutes.
# !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# # Install pre-built detectron2 that matches pytorch version, if released:
# # See https://detectron2.readthedocs.io/tutorials/install.html for instructions
# #!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/{CUDA_VERSION}/{TORCH_VERSION}/index.html
# python3 -m pip install detectron2==0.6 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html

# # exit(0)  # After installation, you may need to "restart runtime" in Colab. This line can also restart runtime
import warnings as wr

wr.filterwarnings("ignore")

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

img = '/home/yln1kor/nikhil-test/cycle.jpg'
im = cv2.imread(img)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# print(outputs["instances"].pred_classes)
# print(outputs["instances"].pred_boxes)
print(outputs['instances'].pred_masks)

pred = [0,2,3,5,6,7,9,10,11]
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
for i in pred:
    out = v.draw_instance_predictions(outputs["instances"][outputs['instances'].pred_classes == i].to("cpu"))
# out = v.draw_instance_predictions(outputs["instances"][outputs['instances'].pred_classes == 0].to("cpu"))
out_path = '/home/yln1kor/nikhil-test/Detectron2/cycle_out.jpg'
cv2.imwrite(out_path,out.get_image())
cv2.imshow("output",out.get_image())
cv2.waitKey(0)
