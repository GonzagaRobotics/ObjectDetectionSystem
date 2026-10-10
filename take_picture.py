import os
from uuid import uuid4

import cv2

vc = cv2.VideoCapture(1)

vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

root_folder = "calibration_folder"

while True:
    rval, frame = vc.read()
    cv2.imshow("preview", cv2.resize(frame, (720, 480)))
    key = cv2.waitKey(50)
    if key == ord('q'):
        break
    if key == ord('p'):
        cv2.imwrite(os.path.join(root_folder, str(uuid4())+".jpg"), frame)