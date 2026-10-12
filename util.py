from typing import Callable
from uuid import uuid4

import cv2
from numpy import save

def testingWebcam(callback: Callable[[cv2.Mat], cv2.Mat], displayVideo: bool = False, webcam_num: int = 0, save_video = False):
    if save_video:
        fourcc = cv2.VideoWriter.fourcc(*'XVID')
        out = cv2.VideoWriter(f'output+{str(uuid4())}.avi', fourcc, 20.0, (1920, 1080))

    if displayVideo:
        cv2.namedWindow("preview")
        resize_size = (1920, 1080) if save_video else (720, 480)
        def frameLoop(frame) -> bool:
            frame = cv2.resize(callback(frame), resize_size)
            cv2.imshow("preview", frame)
            key = cv2.waitKey(50)
            if save_video:
                out.write(frame)
            if key == 27:  # exit on ESC
                cv2.destroyWindow("preview")
                vc.release()
                if save_video:
                    out.release()
                return False
            else:
                return True
    else:

        def frameLoop(frame) -> bool:
            callback(frame)
            return True

    vc = cv2.VideoCapture(webcam_num)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        rval, frame = vc.read()
        if rval:
            if frameLoop(frame) == False:
                break


def testingImage(callback: Callable[[cv2.Mat], cv2.Mat], image: str):
    img = callback((cv2.imread(image))) # type: ignore

    cv2.imshow("Image",cv2.resize(img, (720, 480)))
    while cv2.getWindowProperty("Image", 0) >= 0:
        key = cv2.waitKey(50)
        if key == 27:  # exit on ESC
            cv2.destroyWindow("Image")
            break
