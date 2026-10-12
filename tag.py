import cv2
import numpy as np
import pickle
import pandas as pd

from processor import Processor

class Recognizer:
    def getTagPoints(self, marker_size: float) -> cv2.Mat:
        marker_points = np.array(
            [
                [-marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, -marker_size / 2, 0],
                [-marker_size / 2, -marker_size / 2, 0],
            ],
            dtype=np.float32,
        )
        return cv2.Mat(marker_points)

    def __init__(
        self, pklPath: str = "calibration_alienwarex14r2webcam.pckl", markerSize: float = 0.16
    ):
        # 200 mm size of square tag
        self.objPoints = self.getTagPoints(markerSize)

        with open(pklPath, "rb") as f:
            params = pickle.load(f)
            self.cameraMatrix = params[0]
            self.distCoeffs = params[1]

    def getPose(self, corners: cv2.typing.MatLike):
        return cv2.solvePnP(
            self.objPoints, corners, self.cameraMatrix, self.distCoeffs[0]
        )

    def drawImage(self, rvec, tvec, mat):
        return cv2.drawFrameAxes(mat, self.cameraMatrix, self.distCoeffs[0], rvec, tvec, 1)

class DetectedTag:
    def __init__(self, points: cv2.typing.MatLike, id: int, recognizer: Recognizer):
        self.points = points
        self.id = id

        _, self.rvecs, self.tvecs = recognizer.getPose(corners=points)
        self.X = self.tvecs[0][0]
        self.Y = self.tvecs[1][0]
        self.Z = self.tvecs[2][0]

        self.Pitch = self.rvecs[0][0]
        self.Roll = self.rvecs[1][0]
        self.Yaw = self.rvecs[2][0]

    def createPandasSeries(self) -> pd.Series:
        row = pd.Series()
        row["ID"] = self.id
        row["Pitch"] = self.Pitch
        row["Roll"] = self.Roll
        row["Yaw"] = self.Yaw
        row["X"] = self.X
        row["Y"] = self.Y
        row["Z"] = self.Z
        return row

class TagDetector(Processor):
    predefinedTags = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    def __init__(self, calibration_file: str, marker_size: float):
        self.recognizer = Recognizer(pklPath=calibration_file, markerSize=marker_size)

    def refresh(self, img: cv2.Mat):
        return super().refresh(img)

    def processor(self, mat: cv2.Mat):
        corners, id, _ = cv2.aruco.ArucoDetector(
            dictionary=self.predefinedTags
        ).detectMarkers(image=mat)
        vectors = {}
        r = self.recognizer
        for i in range(len(corners)):
            _, rvec, tvec = r.getPose(corners[i])
            vectors[id[i]] = (tvec, rvec)

        return vectors

    def testingProcessor(self, img: cv2.Mat) -> cv2.Mat:
        corners, id, _ = cv2.aruco.ArucoDetector(
            dictionary=self.predefinedTags
        ).detectMarkers(image=img)
        r = self.recognizer
        for i in range(len(corners)):
            _, rvec, tvec = r.getPose(corners[i])
            tag_detect = DetectedTag(corners[i], id[i], self.recognizer).createPandasSeries()
            print(tag_detect)
            img = r.drawImage(rvec, tvec, img)

        return img
