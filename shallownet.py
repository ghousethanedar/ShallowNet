import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from imutils import paths
from pyimagesearch.preprocess.simplepreprocess import SimplePreprocessor
from sklearn.metrics import classification_report
from pyimagesearch.preprocess.imagetoarraypreprocess import ImageToArrayPreprocessor
from pyimagesearch.dataset.simpledatasetloder import SimpleDatasetLoader
from pyimagesearch.shallownet.arch import ShallowNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD
import warnings
warnings.filterwarnings('ignore')


ap = argparse.ArgumentParser()

ap.add_argument('-d','--dataset',required=True,help='#path to the dataset folder')
ap.add_argument('-o','--output',required=True,help='# path to the model output folder')
#
args = vars(ap.parse_args())

print('[INFO]: Loading images....')
image_paths = list(paths.list_images(args['dataset']))

# Initialize the preprocessors
sp = SimplePreprocessor(32, 32)
itap = ImageToArrayPreprocessor()

# Load the dataset and scale the raw pixel intensities to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, itap])
(data, labels) = sdl.load(image_paths, verbose=500)
data = data.astype('float') / 255.0



# Split the data into training data (75%) and testing data (25%)
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Convert the labels from integers to vectors
train_y = LabelBinarizer().fit_transform(train_y)
test_y = LabelBinarizer().fit_transform(test_y)

# Initialize the optimizer and model
print('[INFO]: Compiling model....')
optimizer = SGD(lr=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the network
print('[INFO]: Training the network....')
H = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=1, epochs=10, verbose=1)

# Test the network
print('[INFO]: Evaluating the network....')
predictions = model.predict(test_x, batch_size=32)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=['cats', 'dogs', 'panda']))



# Plot the training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 10), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 10), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, 10), H.history['accuracy'], label='train_acc')
plt.plot(np.arange(0, 10), H.history['val_accuracy'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()