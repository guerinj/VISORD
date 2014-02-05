from __future__ import division
import cv2
import numpy as np
import os
import ipdb
import math

class Histogram:

    def __init__(self, histResolution, scaleDisplay):
        self.histResolution = histResolution
        self.scaleDisplay = scaleDisplay
        self.histSize = [histResolution, histResolution]
        self.LabRanges = [0, 255, 0, 255]
        self.LabChannels = [1, 2]
        self.LabHist = np.zeros((histResolution, histResolution))

    def addImageToHist(self, image, mask):
        self.LabHist = cv2.calcHist([image], self.LabChannels, mask, self.histSize, self.LabRanges, self.LabHist, True)

    def getDiplayHist(self):
        rawHist = cv2.normalize(self.LabHist)
        displayHist = np.zeros((self.histResolution * self.scaleDisplay, self.histResolution * self.scaleDisplay))
        for a in range(0, 32):
            for b in range(0, 32):
                cv2.rectangle(
                    displayHist,
                    (a*self.scaleDisplay, b*self.scaleDisplay),
                    ((a + 1)*self.scaleDisplay - 1, (b + 1)*self.scaleDisplay - 1),
                    rawHist.item(a, b),
                    cv2.cv.CV_FILLED
                )
        return displayHist


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
    return mask, antimask

peauHistogram = Histogram(32, 10)
nonPeauHistogram = Histogram(32, 10)

#
# First, we scan the dirs to find the test images
#


#
# Below commented lines is for testing only 
#
# dirRawPath = './Raw/'
# dirFacesPath = './Faces/'
# f = "1Person0001.jpg"
# rawImage = cv2.imread(dirRawPath + f)
# faceImage = cv2.imread(dirFacesPath + f.split('.')[0] + '_R.jpg')
# print rawImage.item((1, 1,))
#mask, antimask = generateMask(faceImage)


#print np.array([255, 255, 255])
files = []
dirRawPath = './Raw/'
dirFacesPath = './Faces/'

for dirname, dirnames, filenames in os.walk(dirRawPath):
    for filename in filenames:
        files.append(filename.strip())


#
# We build the histogramms for both kind of images
#

files = files[:1]
cv2.namedWindow("Raw")
cv2.namedWindow("Faces")

for f in files:
    print "Adding file : %s " % dirFacesPath + f.split('.')[0] + '_R.jpg'
    print "Adding file : %s " % dirRawPath + f

    rawImage = cv2.imread(dirRawPath + f)
    faceImage = cv2.imread(dirFacesPath + f.split('.')[0] + '_R.jpg')

    mask, antimask = generateMask(faceImage)

    #cv2.imshow("Raw", mask)
    #cv2.imshow("Faces", antimask)

    nonPeauHistogram.addImageToHist(
        cv2.cvtColor(
            rawImage,
            cv2.COLOR_BGR2LAB),
        antimask
        )

    peauHistogram.addImageToHist(
        cv2.cvtColor(
            faceImage,
            cv2.COLOR_BGR2LAB),
        mask
        )

#
# Below commented lines is for testing only 
#
# dirRawPath = './Raw/'
# dirFacesPath = './Faces/'
# f = "1Person0001.jpg"
# rawImage = cv2.imread(dirRawPath + f)
# faceImage = cv2.imread(dirFacesPath + f.split('.')[0] + '_R.jpg')
# mask, antimask = generateMask(faceImage)


#
# And we display them
#
# cv2.namedWindow("Raw")
# cv2.namedWindow("Faces")
# cv2.imshow("Raw", mask)
# cv2.imshow("Faces", faceImage)

# cv2.waitKey(0)
# cv2.imshow("Raw", antimask)
# cv2.imshow("Faces", rawImage)

# cv2.waitKey(0)
peau = peauHistogram.getDiplayHist()
nonPeau = nonPeauHistogram.getDiplayHist()
print peau
print nonPeau
cv2.waitKey(0)
cv2.imshow("Raw", nonPeauHistogram.getDiplayHist())
cv2.imshow("Faces", peauHistogram.getDiplayHist())

cv2.waitKey(0)
