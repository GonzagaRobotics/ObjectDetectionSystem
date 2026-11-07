from math import sqrt, pow
import pickle
import json

import cv2
import numpy as np

from processor import Processor
from tag import TagDetector

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
    def __init__(self, calibration_file: str, keyboard_consts_file: str):
        with open(keyboard_consts_file, 'r') as file:
            jsonobj = json.load(file)
            self.vecs = {}
            self.marker_size = jsonobj["marker_size"]
            self.CORNERS = {int(k): (float(jsonobj["corners"][k]["coordinates"]["x"]), float(jsonobj["corners"][k]["coordinates"]["y"]), float(jsonobj["corners"][k]["coordinates"]["z"])) for k in jsonobj["corners"]}
            self.CORNER_NAMES = {int(k): jsonobj["corners"][k]["name"] for k in jsonobj["corners"]}
            self.DISTANCES = {}
            for i in jsonobj["distances"]:
                for j in jsonobj["distances"][i]:
                    self.DISTANCES[(int(i), (int(j)))] = ((jsonobj["distances"][i][j]["axis"]), float(jsonobj["distances"][i][j]["direction"]), float(jsonobj["distances"][i][j]["value"]))
            self.tag_processor = TagDetector(calibration_file, self.marker_size)
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
        elif (from_id, to_id) in self.DISTANCES:
            adj[2] = self.CORNERS[to_id][2]-self.CORNERS[from_id][2]
            data = self.DISTANCES[(from_id, to_id)]
            adj[XYZ_MAP[data[0]]] = data[1]*data[-1]
            return np.asarray([adj])
        elif (to_id, from_id) in self.DISTANCES:
            adj[2] = self.CORNERS[from_id][2]-self.CORNERS[to_id][2]
            data = self.DISTANCES[(to_id, from_id)]
            adj[XYZ_MAP[data[0]]] = data[1]*-1*data[-1]
            return np.asarray([adj])
        else:
            mid = list(set(self.CORNERS.keys()).difference((from_id, to_id)))[0]
            return np.asarray(self.get_adj(from_id, mid)) + np.asarray(self.get_adj(mid, to_id))


    def check_keyboard(self, keys: dict):
        for key in self.DISTANCES:
            if self.CORNER_NAMES[key[0]] in keys and self.CORNER_NAMES[key[1]] in keys:
                distance = distance_formula(keys[self.CORNER_NAMES[key[0]]][0], keys[self.CORNER_NAMES[key[1]]][0])
                if abs(distance-self.DISTANCES[key][-1]) > self.TOLERANCE:
                    return False
        return True

    def processor(self, mat: cv2.Mat):
        self.tags = self.tag_processor.processor(mat)
        new_vecs = {}
        for name in self.CORNER_NAMES:
            new_vecs[self.CORNER_NAMES[name]] = [np.asarray([0,0,0]), np.asarray([0,0,0])]
        locks = {x: False for x in self.CORNERS}
        for id in self.tags:
            if id in self.CORNERS and id in self.CORNER_NAMES:
                corner = self.CORNERS[id]
                # correction_factor = self.marker_size/2
                xyz_matrix_corner = np.array([corner[0], corner[1], corner[2]])

                new_rvec = np.asarray(self.tags[id][1])
                tvec = np.asarray(self.tags[id][0]).reshape(3, 1)

                R, _ = cv2.Rodrigues(new_rvec)

                new_tvec = R @ xyz_matrix_corner.reshape(3, 1) + tvec
                new_tvec = new_tvec.reshape(1, 3)

                for to_id in self.CORNERS:
                    if to_id == id:
                        print(new_rvec)
                        new_vecs[self.CORNER_NAMES[to_id]] = [new_tvec[0], new_rvec]
                        locks[to_id] = True
                    elif not locks[to_id]:
                        adj = R @ np.asarray(self.get_adj(id, to_id)+xyz_matrix_corner).reshape(3, 1) + tvec
                        adj = adj.reshape(1, 3)
                        old_tvec = np.asarray(new_vecs[self.CORNER_NAMES[to_id]][0])
                        old_rvec = np.asarray(new_vecs[self.CORNER_NAMES[to_id]][1])
                        tmp_tvec = old_tvec + adj
                        tmp_rvec = old_rvec + new_rvec

                        new_vecs[self.CORNER_NAMES[to_id]] = [np.asarray(tmp_tvec).sum(axis=0), np.asarray(tmp_rvec).sum(axis=0)]

        divisor = sum(list(map(lambda x: x==True, [locks[x] for x in locks])))
        if divisor == 0:
            return {}
        for name in self.CORNER_NAMES:
            if not locks[name]:
                new_vecs[self.CORNER_NAMES[name]][0] = np.asarray(new_vecs[self.CORNER_NAMES[name]][0]) / divisor
                new_vecs[self.CORNER_NAMES[name]][1] = np.asarray(new_vecs[self.CORNER_NAMES[name]][1]) / divisor

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
