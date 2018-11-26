# -*- coding: utf-8 -*-
# code is from [get_class_weights.py](https://github.com/kwotsin/TensorFlow-ENet/blob/master/get_class_weights.py)
import numpy as np
import os
from scipy.misc import imread


def ENet_weighing(image_files, num_classes=12):
    '''
    The custom class weighing function as seen in the ENet paper.
    INPUTS:
    - image_files(list): a list of image_filenames which element can be read immediately
    OUTPUTS:
    - class_weights(list): a list of class weights where each index represents each class label and the element is the class weight for that label.
    '''
    #initialize dictionary with all 0
    label_to_frequency = {}
    for i in xrange(num_classes):
        label_to_frequency[i] = 0

    for n in xrange(len(image_files)):
        image = imread(image_files[n])

        #For each label in each image, sum up the frequency of the label and add it to label_to_frequency dict
        for i in xrange(num_classes):
            class_mask = np.equal(image, i)
            class_mask = class_mask.astype(np.float32)
            class_frequency = np.sum(class_mask)

            label_to_frequency[i] += class_frequency

    #perform the weighing function label-wise and append the label's class weights to class_weights
    class_weights = []
    total_frequency = sum(label_to_frequency.values())
    for label, frequency in label_to_frequency.items():
        class_weight = 1 / np.log(1.02 + (frequency / total_frequency))
        class_weights.append(class_weight)

    #Set the last class_weight to 0.0
    class_weights[-1] = 0.0

    return class_weights

def median_frequency_balancing(image_files, num_classes=12):
    '''
    Perform median frequency balancing on the image files, given by the formula:
    f = Median_freq_c / total_freq_c
    where median_freq_c is the median frequency of the class for all pixels of C that appeared in images
    and total_freq_c is the total number of pixels of c in the total pixels of the images where c appeared.
    INPUTS:
    - image_files(list): a list of image_filenames which element can be read immediately
    - num_classes(int): the number of classes of pixels in all images
    OUTPUTS:
    - class_weights(list): a list of class weights where each index represents each class label and the element is the class weight for that label.
    '''
    #Initialize all the labels key with a list value
    label_to_frequency_dict = {}
    for i in xrange(num_classes):
        label_to_frequency_dict[i] = []

    for n in xrange(len(image_files)):
        image = imread(image_files[n])

        #For each image sum up the frequency of each label in that image and append to the dictionary if frequency is positive.
        for i in xrange(num_classes):
            class_mask = np.equal(image, i)
            class_mask = class_mask.astype(np.float32)
            class_frequency = np.sum(class_mask)

            if class_frequency != 0.0:
                label_to_frequency_dict[i].append(class_frequency)

    class_weights = []

    #Get the total pixels to calculate total_frequency later
    total_pixels = 0
    for frequencies in label_to_frequency_dict.values():
        total_pixels += sum(frequencies)

    for i, j in label_to_frequency_dict.items():
        j = sorted(j) #To obtain the median, we got to sort the frequencies

        median_frequency = np.median(j) / sum(j)
        total_frequency = sum(j) / total_pixels
        median_frequency_balanced = median_frequency / total_frequency
        class_weights.append(median_frequency_balanced)

    #Set the last class_weight to 0.0 as it's the background class
    class_weights[-1] = 0.0

    return class_weights

if __name__ == "__main__":
    image_dir = os.path.expanduser("~/Data/CamVid/trainannot")
    image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.png')]
    print('image_files:', image_files)
    class_weights = median_frequency_balancing(image_files, num_classes=12)
    print('median_frequency_balancing class_weights:', class_weights)
    class_weights = ENet_weighing(image_files, num_classes=12)
    print('ENet_weighing class_weights:', class_weights)
