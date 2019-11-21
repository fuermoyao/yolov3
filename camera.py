#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 17:09:30 2019

@author: wy
"""

from __future__ import division

from utils.utils import *
from models import *
import time
import datetime
import argparse

import torch
from torch.autograd import Variable
import cv2
import numpy as np


def onMouse(event,x,y,flags,param):
    	global clicked
    	if event == cv2.EVENT_LBUTTONUP:
    		clicked = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--usb_camera", type=int, default=0, help="id of usb camera")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))
    model.eval()  # Set in evaluation mode

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    clicked = False    
    cameraCapture = cv2.VideoCapture(opt.usb_camera)
    cameraCapture.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
    cameraCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cameraCapture.set(cv2.CAP_PROP_FPS, 30) 
    success,frame = cameraCapture.read()
    success,frame = cameraCapture.read()
    success,frame = cameraCapture.read()
    time.sleep(1)
    cv2.namedWindow("demo")
    cv2.setMouseCallback("demo",onMouse)
    while success and cv2.waitKey(1) == -1 and not clicked:
        time_start=time.time()
        success,frame = cameraCapture.read()
        frame = cv2.resize(frame,(opt.img_size,opt.img_size))
        input_img = frame[:,:,::-1].transpose((2,0,1))    # BGR -> RGB | H X W C -> C X H X W 
       # input_imgs = cv2.cvtColor(input_imgs, cv2.COLOR_BGR2RGB)
        input_img = input_img[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
        #input_imgs = np.expand_dims(input_imgs, 0) 
        # Configure input
        input_img = torch.from_numpy(input_img)
        input_img = Variable(input_img.type(Tensor))
        input_img = input_img.to(device)
        # Get detections
        with torch.no_grad():
            detections = model(input_img)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
        if detections[0] is not None:
            for i in detections[0]:
               i = i.numpy()
               x1 = int(i[0])
               y1 = int(i[1])
               x2 = int(i[2])
               y2 = int(i[3])
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),3)
               cv2.putText(frame,classes[int(i[-1])],(int(i[0]+20),int(i[1]+40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
            cv2.imshow('demo',frame)
        time_end=time.time()
#        inference_time = datetime.timedelta(seconds=)
        print("\t+ Batch, Inference Time: %s" % (time_end - time_start))

    cameraCapture.release()
    cv2.destroyAllWindows()

   
