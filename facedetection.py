import cv2
import numpy as np
import uuid
import shutil

net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
modell = cv2.dnn.DetectionModel(net)
modell.setInputParams(size=(416, 416), scale=1 / 255)

classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)


cap = cv2.VideoCapture("video1.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#source_file= "video2.mp4"
#file_extansion = source_file.split(".")[1]
#unique_filename= str(uuid.uuid4())
#destination_file = "C:/Users/ronan/OneDrive/Masaüstü/python/"+unique_filename+"."+file_extansion


while True:
    ret, frame = cap.read()

    (class_ids, score, bboxes) = modell.detect(frame)
    for class_ids, score, bbox in zip(class_ids, score, bboxes):
        (x, y, w, h) = bbox

        center_x = int(x + (w // 2))
        center_y = int(y + (h // 2))
        box_x = int(center_x - (300 // 2))
        box_y = int(center_y - (300 // 2))

        class_name = classes[class_ids]
        if class_name == "person":
            cv2.putText(
                frame,
                str(class_name),
                (x, y - 10),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (200, 0, 50),
                2,
            )
            cv2.rectangle(frame, (center_x, center_y),(box_x,box_y), (200, 0, 50), 1)
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
