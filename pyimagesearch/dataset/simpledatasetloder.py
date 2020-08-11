import os
import cv2

class SimpledatasetLoder:
    def __init__(self,preprocessor=None):
        self.preprocessor=preprocessor
        if self.preprocessor is None:
            self.preprocessor = []
    def load(self,imagepaths,verbose=-1):
        data=[]
        labels=[]
        for (i,imagepath) in enumerate(imagepaths):
            label = imagepath.split(os.path.sep)[-2]
            image = cv2.imread(imagepath)

            if self.preprocessor is not None:
                for p in self.preprocessor:
                    p.preprocess()
            data.append(image)
            labels.append(label)

            if verbose>0 and  i>0 and (i+1)%verbose==0:
                print("[INFO] Loading images...{}/{}".format(i+1,len(imagepaths)))
        return (data,label)



