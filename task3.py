import cv2
import glob
import os
import uuid
import numpy

videos_path = glob.glob("*.mp4")
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
modell = cv2.dnn.DetectionModel(net)
modell.setInputParams(size=(416, 416), scale=1 / 255)

classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)


for video_path in sorted(videos_path):
    print("Processing video: ", video_path)
    cap = cv2.VideoCapture(video_path)

    counter = 0
    save_folder = video_path.split(".")[0]
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    while True:
        ret, frame = cap.read()
        counter += 1

        if not ret:
            break

        if counter % 5 == 0:
            height = frame.shape[0]
            width = frame.shape[1]
            center_x = width // 2
            center_y = height // 2
            w = 300
            h = 300
            new_frame = frame[center_y : center_y + h, center_x : center_x + w]
          
            (class_ids, score, bboxes) = modell.detect(new_frame)
            if class_ids == 0:
                new1_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(f"{save_folder}/{str(uuid.uuid4())}.jpg", new1_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    cap.release()
    cv2.destroyAllWindows()
