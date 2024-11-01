import cv2
import numpy as np
import pickle
import pandas as pd

class Recognizer():
    def getTagPoints(self, marker_size: float) -> cv2.Mat:
        marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                  [marker_size / 2, marker_size / 2, 0],
                                  [marker_size / 2, -marker_size / 2, 0],
                                  [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
        return marker_points

    def __init__(self, pklPath: str = "../data/calibration_alienwarex14r2webcam.pckl", markerSize: float = 0.002):
        # 200 mm size of square tag
        self.objPoints = self.getTagPoints(markerSize)

        with open(pklPath, 'rb') as f:
            params = pickle.load(f)
            self.cameraMatrix = params[0]
            self.distCoeffs = params[1]

    def getPose(self, corners: cv2.typing.MatLike):
        return cv2.solvePnP(self.objPoints, corners, self.cameraMatrix, self.distCoeffs[0])

class DetectedTag():
    def __init__(self, points: cv2.typing.MatLike, id: int):
        self.points = points
        self.id = id
        totalX = 0
        totalY = 0
        for point in points:
            totalX = totalX + point[0]
            totalY = totalY + point[1]
        self.centerX = int(totalX / 4)
        self.centerY = int(totalY / 4)

        _, self.rvecs, self.tvecs = Recognizer().getPose(corners=points)
        self.X = self.tvecs[0][0]
        self.Y = self.tvecs[1][0]
        self.Z = self.tvecs[2][0]

        self.Pitch = self.rvecs[0][0]
        self.Roll = self.rvecs[1][0]
        self.Yaw = self.rvecs[2][0]

    def createPandasSeries(self) -> pd.Series:
        row = pd.Series()
        row['ID'] = self.id
        row['Pitch'] = self.Pitch
        row['Roll'] = self.Roll
        row['Yaw'] = self.Yaw
        row['X'] = self.X
        row['Y'] = self.Y
        row['Z'] = self.Z
        row['centerX'] = self.centerX
        row['centerY'] = self.centerY
        return row

class TagDetector():
    predefinedTags = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    def processor(self, mat: cv2.Mat) -> pd.DataFrame:
        r = Recognizer()
        corners, id, _ = cv2.aruco.ArucoDetector(dictionary=self.predefinedTags).detectMarkers(image=mat)
        tagsPd = pd.DataFrame()
        if (len(corners) > 0):
            for i in range(0, len(corners)):
                tag =  (DetectedTag(points=corners[i][0], id=id[i][0]))
                tmp = pd.DataFrame([DetectedTag(points=corners[i][0], id=id[i][0]).createPandasSeries().to_dict()])
                tagsPd = pd.concat([tagsPd, tmp])
        return tagsPd

    def testingProcessor(self, tDf: pd.DataFrame, img: cv2.Mat) -> cv2.Mat:
        for index, row in tDf.iterrows():
            img = cv2.circle(img, (int(row['centerX']), int(row['centerY'])), 10, (0, 255, 255), -1)
        return img