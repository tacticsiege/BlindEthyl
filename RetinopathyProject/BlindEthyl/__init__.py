import cv2
import os
import numpy as np
import csv
from os import listdir
from os.path import isfile, join


def load_image(path):
    image = cv2.imread(path)
    return image


def load_grayscale_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def load_images_raw(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    images = dict()
    for filename in files:
        key = filename[0:len(filename)-5]
        images[key] = load_image(join(path, filename))

    return images


def load_grayscale_images(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    images = dict()
    for filename in files:
        key = filename[0:len(filename)-5]
        images[key] = load_grayscale_image(join(path, filename))
    return images


def load_labels(path):
    filename = join(path, 'trainLabels.csv')
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        labels = list(reader)
    return dict((r[0], r[1]) for r in labels[1:])


def load_grayscale_dataset(image_path, label_path):
    labels = load_labels(label_path)
    images = load_grayscale_images(image_path)

    # combine label and image and flatten
    dataset = dict((key, (labels[key], images[key].flatten()))
                   for key in images.iterkeys())

    X_train = []
    y_train = []
    for d in dataset.iterkeys():
        X_train.append(dataset[d][1])
        y_train.append(dataset[d][0])

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    return X_train, y_train
