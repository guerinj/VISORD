from __future__ import division
import cv2
import numpy as np
import os
import ipdb
import math
from ipdb import launch_ipdb_on_exception


###############################
#  UTILS
###############################

def generateMask(image):
    antimask = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
    mask = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)

    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            if math.sqrt(image.item(x, y, 0)**2 + image.item(x, y, 1)**2 + image.item(x, y, 2)**2) > math.sqrt(3*250**2):
                mask.itemset((x, y, 0), 255)
                antimask.itemset((x, y, 0), 0)
            else:
                mask.itemset((x, y, 0), 0)
                antimask.itemset((x, y, 0), 255)
    return antimask, mask

if __name__ == "__main__":
    #
    # First, we scan the dirs to find the test images
    #

    files = []
    dirRawPath = './Raw/'
    dirFacesPath = './Faces/'

    for dirname, dirnames, filenames in os.walk(dirRawPath):
        for filename in filenames:
            files.append(filename.strip())

    #cv2.namedWindow("Original")
    #cv2.namedWindow("Mask")
    #
    # We build the histogramms for both kind of images
    #
    histRange = 32
    scale = 10
    # labHistPeau = Histogram(histRange, scale, "LAB")
    # rgbHistPeau = Histogram(histRange, scale, "RGB")
    # hsvHistPeau = Histogram(histRange, scale, "HSV")

    # labHist = Histogram(histRange, scale, "LAB")
    # rgbHist = Histogram(histRange, scale, "RGB")
    # hsvHist = Histogram(histRange, scale, "HSV")

    files = files[:10]
    for f in files:
        print "Adding file : %s " % dirFacesPath + f.split('.')[0] + '_R.jpg'
        print "Adding file : %s " % dirRawPath + f

        rawImage = cv2.imread(dirRawPath + f)
        faceImage = cv2.imread(dirFacesPath + f.split('.')[0] + '_R.jpg')

        mask, antimask = generateMask(faceImage)

        
        #labHist.addImageToHist(
        #    rawImage,
        #    antimask
        #    )
        cv2.imshow("Original", rawImage)
        cv2.imshow("Mask", antimask)
        cv2.waitKey(0)
        
        #labHistPeau.addImageToHist(
        #    faceImage,
        #    mask
        #    )
        
        cv2.imshow("Original", faceImage)
        cv2.imshow("Mask", mask)
        cv2.waitKey(0)