import cv2
from tensorflow.keras.preprocessing.image import img_to_array


class ImageToArrayPreProcess:
    def __init__(self,dataFormat=None):
        self.dataFormat=dataFormat

    def preprocess(self, image):
        image=img_to_array(image,data_format=self.dataFormat)
        return image
