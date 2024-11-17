import os

import pandas as pd

from inference import ObjDetector
from tag import TagDetector
import cv2
import copy

from util import testingWebcam

t = TagDetector()
# i = ObjDetector()

class Processor:
    def refresh(self, img: cv2.Mat):
        self.tResult = t.processor(img)
        # self.iResult = i.processor(img)

    def processor(self, img: cv2.Mat) -> pd.DataFrame:
        return self.tResult.reset_index().drop(axis='columns', columns=['index'])

    def testingProcessor(self, img: cv2.Mat) -> cv2.Mat:
        self.refresh(img)
        print(self.tResult.reset_index().drop(axis='columns', columns=['index']))
        print("\r", sep='')
        newImg = t.testingProcessor(self.tResult, copy.deepcopy(img))
        # newImg = i.testingProcessor(self.iResult, newImg)
        return newImg

p = Processor()
testingWebcam(p.testingProcessor, True)