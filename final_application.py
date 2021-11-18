# Next: https://towardsdatascience.com/video-streaming-in-web-browsers-with-opencv-flask-93a38846fe00

import sys
import os
import cv2
from utils import get_outputs_names, post_process

import torch
from model import CNN
import numpy as np
from matplotlib import pyplot as plt

# Constants
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416

# Params to change
offset = 35

# Model inputs
model_cfg = r"C:/Users/andre\Desktop/Facultatea alternativa de inteligenta/Modulul 8/proiectComplet/codEu/yoloface/cfg/yolov3-face.cfg"
model_weights = r"C:/Users/andre/Desktop/Facultatea alternativa de inteligenta/Modulul 8/proiectComplet/codEu/yoloface/model-weights/yolov3-wider_16000.weights"

# Load the model #
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Classification models args #
res = (64, 64)
model_class_weights = r"C:/Users/andre\Desktop/Facultatea alternativa de inteligenta/Modulul 8/proiectComplet/codEu/yoloface/Epoch79_Error0.02"
###############

# Load the classification model #
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device used for classification model: {device}")
model = CNN().to(device)
model.load_state_dict(torch.load(model_class_weights))
model.eval() 
##################

# Added methods for classification #
def preprocess_frame(frame):
    """
    1. Convert to grayscale
    2. Resize
    3. Normalize
    4. Convert to float32 (double)
    5. Add channel dim + batch dim !! (nbatch, nchannel, widht, height)
    6. Convert to torch tensor
    """
    frame = rgb2gray(frame)
    frame = cv2.resize(frame, res)
    frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
    frame = frame.astype(np.float32)
    frame = np.reshape(frame, (1, 1, frame.shape[0], frame.shape[1]))
    frame = torch.tensor(frame)
    return frame

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
##################################################


# Stream capturer
cap = cv2.VideoCapture(0)
while True:
    has_frame, frame = cap.read()

    # Delay in miliseconds between every frame and wait for a key press
    key = cv2.waitKey(1) & 0xFF

    # YOLO PART #
    # Create a 4D blob from a frame.
    # More info: https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
    # It does standardization
    # Image, scaleFactor for 0-1 norm, new size, swapRB (opencv asteapta imagini in BGR, dar imaginea e RGB)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                [0, 0, 0], 1, crop=False)
        
    net.setInput(blob)
    outs = net.forward(get_outputs_names(net))
    faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD) # [[leftTop_x, leftTop_y, width, height]]

    if faces:
        # Extract points
        leftTopX = faces[0][0]
        leftTopY = faces[0][1]

        rightBottomX = faces[0][0] + faces[0][2] # (leftTop_x + width)
        rightBottomY = faces[0][1] + faces[0][3] # (leftTop_y + height)

        # Facut crop la fata
        my_face = frame[leftTopY - offset: rightBottomY + offset, leftTopX - offset: rightBottomX + offset]
        
        if 0 not in my_face.shape:
            # Classify face #
            prep_face = preprocess_frame(my_face)
            pred = model(prep_face)
            pred = torch.max(pred, 1)[1].data.squeeze().item()
            
            predText = "No Mask" if pred==0 else ("MaskWrong" if pred==1 else "Mask") 
            color_pred = (0, 0, 255) if pred==0 else ((0, 255, 255) if pred==1 else (0, 255, 0)) 
            #################
            
            # Desenat rectangle
            cv2.rectangle(frame, (leftTopX, leftTopY), (rightBottomX, rightBottomY), color=color_pred, thickness=3)
            cv2.putText(frame, predText, (leftTopX, leftTopY), cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=color_pred, thickness=3)
        #############

    if key == ord("q"):
        break
    cv2.imshow('Video stream', frame)

# After the loop release the cap object and destroy all windows
cap.release()
cv2.destroyAllWindows()
