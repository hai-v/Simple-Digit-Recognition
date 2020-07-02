#%%
import sys
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import random

import os
import numpy as np
from sklearn.utils import shuffle

### Constants ###
numDigits = 10
imageSize = 32
originalFolder = "originalSamples"
cleanFolder = "cleanSamples"
filePrefix = "digit_"
fileExtension = ".png"
studentNumbers = [
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "13",
    "14", "16", "18", "19", "20", "21", "22"
]  # missing samples 12, 15, 17
classifiers = {
    "Gaussian - Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(gamma='scale'),
    "Decision Tree": DecisionTreeClassifier(),
    "K Nearest Neighbors": KNeighborsClassifier()
}


# Make all samples uniformly black-on-white.
# For each original sample, make a copy where all non-white pixels are black.
# Save modified samples to [cleanFolder]
def cleanFiles():
    os.mkdir("./" + cleanFolder)
    for i in range(numDigits):
        for j in studentNumbers:
            fileName = "digit_" + str(i) + "_" + j + fileExtension
            img = Image.open("./" + originalFolder + "/" + fileName)
            width, height = img.size
            for h in range(height):
                for w in range(width):    
                    pixel = img.getpixel((w, h))
                    if pixel[0:3] != (255,255,255):
                        img.putpixel((w, h), (0,0,0))
            img.save("./" + cleanFolder + "/" + fileName)


# Create images and targets.
# For each clean sample, the image should be a 32 x 32 binary matrix
#     where if the pixel is white, put 0, if black, put 1
# Targets should be a one-dimensional array containing the digits 
#     that each sample represents.
def buildImagesAndTargets():
    images = []
    targets = []
    for sn in studentNumbers:
        for nd in range(numDigits):
            fileName = "digit_" + str(nd) + "_" + sn + fileExtension
            img = Image.open("./" + cleanFolder + "/" + fileName)
            imgH = []
            width,height = img.size
            for h in range(height):
                imgW = []
                for w in range(width):
                    pixel = img.getpixel((w, h))
                    if pixel[0:3] == (255, 255, 255):
                        imgW.append(0)
                    else:
                        imgW.append(1)
                imgH.append(imgW)
            images.append(imgH)
            targets.append(nd)
    return images, targets


# Convert each image to a one-dimensional array
def flatten(images):
    newImages = []
    for i in images:
        image = []
        for h in range(len(i)):
            for w in range(len(i[h])):
                image.append(i[h][w])
        newImages.append(image)
    return newImages 


# Run the classifier comparison code (a la A8).
# For each classifier, show the classification report and the confusion matrix.
# At the end show the four accuracy scores together in a summary section.
def trainPredictReport(images, targets):
    summary = "--------Summary--------" + "\n"
    for c in classifiers:
        X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=0.5, shuffle=False)
        classifiers[c].fit(X_train, y_train)
        predicted = classifiers[c].predict(X_test)
        classification_report = metrics.classification_report(y_test, predicted, output_dict=True)
        print("Classification report for classifier %s:\n%s\n" % (c, metrics.classification_report(y_test, predicted)))
        disp = metrics.plot_confusion_matrix(classifiers[c], X_test, y_test)
        print("Confusion matrix:\n%s" % disp.confusion_matrix)
        summary += c + "\t" + str(int(classification_report['accuracy'] * 100)) + "%\n"
    print(summary)


# Shuffle the images and targets in the same order.
def shuffleImagesAndTargets(images, targets):
    images, targets = np.array(images), np.array(targets)
    images, targets = shuffle(images, targets)
    return images, targets


# Convert each image from a 32-by-32 binary matrix into
# a 8-by-8 matrix where each cell is the number of 1's
# in each 4-by-4 block of the original matrix.
# (Same as the built-in dataset)
def reduceDimensions(images):
    newImages = []
    for i in images:
        imgH = []
        for h in range(8):
            imgW = []
            for w in range(8):
                count = 0
                for y in range(4):
                    for x in range(4):
                        if i[y + h * 4][x + w * 4] == 1:
                            count = count + 1
                imgW.append(count)
            imgH.append(imgW)
        newImages.append(imgH)
    return newImages

# Mix our data with samples from the built-in dataset.

def mixWithBuiltins(images, targets, numSamples):
    targets = targets.tolist()
    digits = datasets.load_digits()
    data = digits.images.reshape((len(digits.images), -1))
    for i in range(numSamples):
        images.append(data[i])
        targets.append(digits.target[i])
    return images, targets

def main():
    images, targets = buildImagesAndTargets()
    images = reduceDimensions(images)
    images, targets = shuffleImagesAndTargets(images, targets)
    images = flatten(images)
    images, targets = mixWithBuiltins(images, targets, 100)
    trainPredictReport(images, targets)


### Main Driver ###
if __name__ == '__main__':
    main()

# %%
