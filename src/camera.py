import cv2
import uuid

video = cv2.VideoCapture(1)

while True:
    _, frame = video.read()
    cv2.imshow("preview", frame)
    key = cv2.waitKey(5)
    if key == ord('c'):
        cv2.imwrite(str(uuid.uuid4()) + ".jpg", frame)
        print("saved")
    elif key == 27:
        break