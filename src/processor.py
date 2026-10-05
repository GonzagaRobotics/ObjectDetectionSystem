from abc import ABC, abstractmethod
import copy

import cv2
import pandas as pd


class Processor(ABC):
    @abstractmethod
    def refresh(self, img: cv2.Mat):
        pass

    @abstractmethod
    def processor(self, img: cv2.Mat) -> pd.DataFrame:
        pass

    @abstractmethod
    def testingProcessor(self, img: cv2.Mat) -> cv2.Mat:
        pass