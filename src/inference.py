import cv2
import pandas as pd
import torch
from ultralytics import YOLO

class DetectedObj:
    def __init__(self, t: torch.Tensor, i: torch.Tensor, dictionary: list):
        self.id = dictionary[int(torch.Tensor.cpu(i).numpy().astype(int).tolist()[0])]
        self.xyxy = torch.Tensor.cpu(t).numpy().astype(int).tolist()
        self.series = pd.DataFrame({"ID": [self.id], "X1": [self.xyxy[0]], "Y1": [self.xyxy[1]], "X2": [self.xyxy[2]], "Y2": [self.xyxy[3]]})


class ObjDetector:
    def __init__(self):
        self.model = YOLO("../data/yolo11x.pt")

    def processor(self, image: cv2.Mat) -> pd.DataFrame:
        tDf = pd.DataFrame()
        outputs = self.model.predict(image, conf=0.1, device="cuda:0", verbose=False)
        for output in outputs:
            for box in output.boxes:
                tmp = DetectedObj(box.xyxy[0], box.cls, self.model.names).series
                tDf = pd.concat([tDf, tmp])
        return tDf

    def testingProcessor(self, tDf: pd.DataFrame, image: cv2.Mat) -> cv2.Mat:
        for index, row in tDf.iterrows():
            image = cv2.rectangle(image, (row['X1'], row['Y1']), (row['X2'], row['Y2']), (255, 0, 0), 10)
            image = cv2.putText(image, row['ID'], (row['X1'], row['Y1']-5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))
        return image
