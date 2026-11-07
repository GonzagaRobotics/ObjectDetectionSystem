from math import pi, sqrt, pow
import pickle

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
    (0, 3): ('y', -1, 12.38),
    (1, 2): ('y', -1, 12.38),
    (2, 3): ('x', 1, 35.24),
    (1, 0): ('x', 1, 35.24)
}

XYZ_MAP = {
    'x': 0,
    'y': 1,
    'z': 2
}

def distance_formula(a, b):
    total = 0
    for i in range(len(a)):
        total += pow(a[i]-b[i], 2)
    return sqrt(total)

class KeyboardPose(Processor):
    def __init__(self, calibration_file: str, marker_size: float):
        self.vecs = {}
        self.marker_size = marker_size
        self.tag_processor = TagDetector(calibration_file, marker_size)
        self.recognizer = self.tag_processor.recognizer
        with open(calibration_file, "rb") as f:
            params = pickle.load(f)
            self.cameraMatrix = params[0]
            self.distCoeffs = params[1]


    def refresh(self, img: cv2.Mat):
        return super().refresh(img)

    # in cm
    TOLERANCE = 0.75

    def get_adj(self, from_id, to_id):
        factor = 0
        adj = [0.0,0.0,0.0]
        if from_id == to_id:
            return np.asarray([adj])
        elif (from_id, to_id) in DISTANCES:
            adj[2] = CORNERS[to_id][2]-CORNERS[from_id][2]
            data = DISTANCES[(from_id, to_id)]
            adj[XYZ_MAP[data[0]]] = data[1]*data[-1]
            return np.asarray([adj])
        elif (to_id, from_id) in DISTANCES:
            adj[2] = CORNERS[from_id][2]-CORNERS[to_id][2]
            data = DISTANCES[(to_id, from_id)]
            adj[XYZ_MAP[data[0]]] = data[1]*-1*data[-1]
            return np.asarray([adj])
        else:
            mid = list(set(CORNERS.keys()).difference((from_id, to_id)))[0]
            return np.asarray(self.get_adj(from_id, mid)) + np.asarray(self.get_adj(mid, to_id))


    def check_keyboard(self, keys: dict):
        for key in DISTANCES:
            if CORNER_NAMES[key[0]] in keys and CORNER_NAMES[key[1]] in keys:
                distance = distance_formula(keys[CORNER_NAMES[key[0]]][0], keys[CORNER_NAMES[key[1]]][0])
                if abs(distance-DISTANCES[key][-1]) > self.TOLERANCE:
                    return False
        return True

    def processor(self, mat: cv2.Mat):
        self.tags = self.tag_processor.processor(mat)
        new_vecs = {}
        for name in CORNER_NAMES:
            new_vecs[CORNER_NAMES[name]] = [np.asarray([0,0,0]), np.asarray([0,0,0])]
        locks = {x: False for x in CORNERS}
        for id in self.tags:
            if id in CORNERS and id in CORNER_NAMES:
                corner = CORNERS[id]
                # correction_factor = self.marker_size/2
                xyz_matrix_corner = np.array([corner[0], corner[1], corner[2]])

                new_rvec = np.asarray(self.tags[id][1])
                tvec = np.asarray(self.tags[id][0]).reshape(3, 1)

                R, _ = cv2.Rodrigues(new_rvec)

                new_tvec = R @ xyz_matrix_corner.reshape(3, 1) + tvec
                new_tvec = new_tvec.reshape(1, 3)

                for to_id in CORNERS:
                    if to_id == id:
                        print(new_rvec)
                        new_vecs[CORNER_NAMES[to_id]] = [new_tvec[0], new_rvec]
                        locks[to_id] = True
                    elif not locks[to_id]:
                        adj = R @ np.asarray(self.get_adj(id, to_id)+xyz_matrix_corner).reshape(3, 1) + tvec
                        adj = adj.reshape(1, 3)
                        old_tvec = np.asarray(new_vecs[CORNER_NAMES[to_id]][0])
                        old_rvec = np.asarray(new_vecs[CORNER_NAMES[to_id]][1])
                        tmp_tvec = old_tvec + adj
                        tmp_rvec = old_rvec + new_rvec

                        new_vecs[CORNER_NAMES[to_id]] = [np.asarray(tmp_tvec).sum(axis=0), np.asarray(tmp_rvec).sum(axis=0)]

        divisor = sum(list(map(lambda x: x==True, [locks[x] for x in locks])))
        if divisor == 0:
            return {}
        for name in CORNER_NAMES:
            if not locks[name]:
                new_vecs[CORNER_NAMES[name]][0] = np.asarray(new_vecs[CORNER_NAMES[name]][0]) / divisor
                new_vecs[CORNER_NAMES[name]][1] = np.asarray(new_vecs[CORNER_NAMES[name]][1]) / divisor

        if self.check_keyboard(new_vecs):
            self.vecs = new_vecs

        return self.vecs

    def testingProcessor(self, img: cv2.Mat) -> cv2.Mat:
        result = self.processor(img)
        if result:
            for name in result:
                rvec = result[name][1]
                tvec = result[name][0]
                img = self.recognizer.drawImage(rvec, tvec, img)
        # return self.tag_processor.testingProcessor(img)
        return img
