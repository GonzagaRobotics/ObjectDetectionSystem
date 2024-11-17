from typing import Callable
import cv2

def testingWebcam(callback: Callable[[cv2.Mat], cv2.Mat], displayVideo: bool = False):
    if displayVideo:
        cv2.namedWindow("preview")

        def frameLoop(frame) -> bool:
            cv2.imshow("preview", callback(frame))
            key = cv2.waitKey(50)
            if key == 27:  # exit on ESC
                cv2.destroyWindow("preview")
                vc.release()
                return False
            else:
                return True
    else:
        def frameLoop(frame) -> bool:
            callback(frame)
            return True

    vc = cv2.VideoCapture(0)
    while True:
        rval, frame = vc.read()
        if rval:
            if frameLoop(frame) == False:
                break