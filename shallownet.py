import argparse
import os
from imutils import paths
from pyimagesearch.preprocess.simplepreprocess import SimplePreprocess
from pyimagesearch.preprocess.imagetoarraypreprocess import ImageToArrayPreProcess
from pyimagesearch.dataset.simpledatasetloder import SimpledatasetLoder
from pyimagesearch.shallownet.arch import Shallownet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import SGD
import warnings
warnings.filterwarnings('ignore')


ap = argparse.ArgumentParser()

ap.add_argument('-d','--dataset',required=True,help='#path to the dataset folder')
ap.add_argument('-o','--output',required=True,help='# path to the model output folder')

args = vars(ap.parse_args())

sp =SimplePreprocess(32,32)
iap = ImageToArrayPreProcess()
sdl = SimpledatasetLoder(preprocessor=[sp,iap])

imagepaths=list(paths.list_images(args['dataset']))

#print(imagepaths)

(data,lables)=sdl.load(imagepaths,verbose=5)

print(data)
#
# print(lables)
data = data.astype("float") / 255.0

le = LabelEncoder()

labels=le.fit_transform(lables)
(trainX,testX,trainy,testy) = train_test_split(data,lables,test_size=0.2,random_state=123)

model = Shallownet.build(height=32,width=32,depth=3,classes=3)
model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.05),metrics=['accuracy'])


