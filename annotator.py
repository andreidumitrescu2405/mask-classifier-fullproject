import sys
import os
import cv2
from utils import get_outputs_names, post_process
import secrets

# Constants
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416

images_acquired = 0

# Params to change
out_path = r"E:\HOBBY2PROFIT\Curs\Editia 1\Modul 6\codEu\yoloface\faces_dataset\mask"
offset = 35

# Model inputs
model_cfg = './cfg/yolov3-face.cfg'
model_weights = './model-weights/yolov3-wider_16000.weights'

# Load the model
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

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
        # Desenat rectangle
        # (leftTop_x, leftTop_y), (rightBottomX, rightBottomY)
        # cv2.rectangle(frame, (leftTopX, leftTopY), (rightBottomX, rightBottomY), color=(0, 255, 0), thickness=3)
        #############

        # Check if crop is done fine
        if 0 not in my_face.shape:
            cv2.imshow('Cropped face', my_face)

            # Save image
            # characte to unicode (standard encodare)
            if key == ord("r"):
                cv2.imwrite(os.path.join(out_path, secrets.token_hex(5)+".png"), my_face)
                images_acquired+=1
                print(f"Captured {images_acquired} images this session!")

    if key == ord("q"):
        break
    cv2.imshow('Video stream', frame)

# After the loop release the cap object and destroy all windows
cap.release()
cv2.destroyAllWindows()
