import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]/'yolov5'  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox
from ament_index_python.packages import get_package_share_directory

model_config = get_package_share_directory('yolov5_ros') + '/config/coco128.yaml'

@torch.no_grad()
class Yolov5:
    def __init__(
        self,
        weights=ROOT/'yolov5s.pt',  # model.pt path(s)
        data=model_config,  # dataset.yaml path
        imgsz=[640, 640],  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=True,  # class-agnostic NMS
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
    ):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det

        self.device = select_device(str(device))
        self.model = DetectMultiBackend(
            weights, 
            device=self.device, 
            dnn=True, 
            data=data
        )
        
        self.stride, self.names, self.pt, jit, onnx, engine = (
            self.model.stride,
            self.model.names,
            self.model.pt,
            self.model.jit,
            self.model.onnx,
            self.model.engine,
        )

        # Setting inference size
        self.img_size = imgsz
        self.img_size = check_img_size(self.img_size, s=self.stride)

        # Half
        self.half = False #HERE
        self.half &= (
            self.pt or jit or onnx or engine
        ) and self.device.type != "cpu"  # FP16 supported on limited backends with CUDA
        if self.pt or jit:
            self.model.model.half() if self.half else self.model.model.float()
        bs = 1  # batch_size
        cudnn.benchmark = True  # set True to speed up constant image size inference


    def predict(self, im):
        im, im0 = self.preprocess(im)

        # Run inference
        im = torch.from_numpy(im).to(self.device) 
        im = im.half() if self.half else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        classes = None
        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(
            pred, 
            self.conf_thres, 
            self.iou_thres, 
            classes, 
            self.agnostic_nms, 
            max_det=self.max_det
        )
        
        # Process predictions 
        det = pred[0].cpu().numpy()

        classes = []
        bounding_boxes = []
        confidence = []
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                box = []
                box.append((xyxy[0], xyxy[1]))
                box.append((xyxy[2],xyxy[3]))
                object_index = int(cls)
                bounding_boxes.append(box)
                classes.append(self.names[object_index])
                confidence.append(conf)

        return classes, bounding_boxes, confidence


    def preprocess(self, im):
        """
        Adapted from yolov5/utils/datasets.py LoadStreams class
        """
        im0 = im.copy()
        im = np.array([letterbox(im, self.img_size, stride=self.stride, auto=self.pt)[0]])
        # Convert
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        im = np.ascontiguousarray(im)

        return im, im0 
