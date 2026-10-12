import pickle
from unittest import result

import cv2
import numpy as np

from processor import Processor
from tag import TagDetector

CORNERS = {
    0: (-0.32, -1.88, 2.76),
    3: (-0.1, 2.025, 2.18),
    2: (0.16, 2.478, 2.18),
    1: (-0.254, -2.6, 2.76)
}

CORNER_NAMES = {
    0: "top_right_corner",
    3: "bottom_right_corner",
    2: "bottom_left_corner",
    1: "top_left_corner"
}

DISTANCES = {
    (0, 3): 12.3
}

class KeyboardPose(Processor):
    def __init__(self, calibration_file: str, marker_size: float):
        self.marker_size = marker_size
        self.tag_processor = TagDetector(calibration_file, marker_size)
        self.recognizer = self.tag_processor.recognizer
        with open(calibration_file, "rb") as f:
            params = pickle.load(f)
            self.cameraMatrix = params[0]
            self.distCoeffs = params[1]


    def refresh(self, img: cv2.Mat):
        return super().refresh(img)

    # def check_keyboard_constraints(self, vecs: dict) -> bool:
    #     for key in DISTANCES:
    #         if (all([CORNER_NAMES[x] in vecs for x in key])):


    #     return True

    def processor(self, mat: cv2.Mat):
        self.tags = self.tag_processor.processor(mat)
        new_vecs = {}
        for id in self.tags:
            if id in CORNERS and id in CORNER_NAMES:
                corner = CORNERS[id]
                correction_factor = self.marker_size/2
                xyz_matrix_corner = np.array([corner[0], corner[1], corner[2]])

                rvec = self.tags[id][1]
                tvec = np.asarray(self.tags[id][0]).reshape(3, 1)

                R, _ = cv2.Rodrigues(rvec)

                new_tvec = R @ xyz_matrix_corner.reshape(3, 1) + tvec

                new_vecs[CORNER_NAMES[id]] = (new_tvec, rvec)
        return new_vecs

    def testingProcessor(self, img: cv2.Mat) -> cv2.Mat:
        result = self.processor(img)
        for name in result:
            rvec = result[name][1]
            tvec = result[name][0]
            img = self.recognizer.drawImage(rvec, tvec, img)
        # return self.tag_processor.testingProcessor(img)
        return img
