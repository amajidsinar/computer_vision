from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from pyimagesearch.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input dataset")
ap.add_argument("-o", "--output", required=True, help="path to output HDF5 file")
ap.add_argument("-b", "--batch_size", type=int, default=32, help="batch size of images to be passed through network")
ap.add_argument("-s", "--buffer_size", type=int, default=1000, help="size of feature extraction buffer")
args = vars(ap.parse_args())            
           
# store the batch size in a convenient variable
bs = args["batch_size"]

# grab the list of images that we'll be describing randomly
# shuffle them to allow for easy training and testing splits via
# array slicing during training time

print("[INFO] loading images ...]")
images = list(paths.list_images(args["input"]))
random.shuffle(images)

# extract the class labels from the image paths then encode the labels
# with the assumsion that the path have the structure of
# dataset_name/{class_label}/example.jpg
labels = [p.split("/")[-2] for p in images]
le = LabelEncoder()
labels = le.fit_transform(labels)

# load the VGG16 network
print("[INFO] loading network ...")
# the last FCC layer should not be included in the architecture
model = VGG16(weights="imagenet", include_top=False)

# initialize the HDF5 dataset writer, then store the class label names in the dataset

# 512 * 7 * 7 arises from the fact that before the final FC layer, max pooling layer have the dimension of 512,7,7 in VGG
dataset = HDF5DatasetWriter((len(images), 512 * 7 *7), args["output"])


# initialize the progress bar
widgets = ["Extracting features: ", progressbar.Percentage(), " ",
           progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(images), widgets=widgets).start()

for i in np.arange(0, len(images), bs):
    # make batches of extraction, e.g [0,1000,2000,3000]
    batchImages = images[i: (i+bs)]
    batchLabels = labels[i: (i+bs)]
    batchPixels = []
    
    # for each batch
    for image in batchImages:
        # load the input image using the Keras helper utility
        # while ensuring te image is resized to 224x224 pixels
        pixels = load_img(image, target_size=(224,224))
        pixels = img_to_array(pixels)

        # preprocess the image by 
        #(1) expanding the dimensions
        pixels = np.expand_dims(pixels, axis=0)
        #(2) substracting the mean RGB pixel intensity
        pixels = imagenet_utils.preprocess_input(pixels)
        
        # add the image to the batch
        batchPixels.append(pixels)
        
        # pass the images through the network and use the outputs as our
        # actual features
    batchPixels = np.vstack(batchPixels)
    features = model.predict(batchPixels, batch_size=bs)

    # reshape the features so that each image is represented by a flattened feature 
    # vector of the 'MaxPooling2D' outputs
    features = features.reshape((features.shape[0], 512 * 7 * 7))

    # add the features and labels to our HDF5 dataset
    dataset.accumulate(features, batchLabels)
    pbar.update(i)

# close the dataset
dataset.close()
pbar.finish()
    
