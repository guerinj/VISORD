from __future__ import division
import cv2
import numpy as np
import os
import ipdb
import math
from ipdb import launch_ipdb_on_exception


#########################
# Histogram classes
#########################

class Histogram:

    def __init__(self, histResolution, scaleDisplay):
        self.histResolution = histResolution
        self.scaleDisplay = scaleDisplay
        self.histSize = [histResolution, histResolution]
        self.Ranges = [0, 255, 0, 255]
        self.Channels = [1, 2]
        self.Hist = np.zeros((histResolution, histResolution))

    def addImageToHist(self, image, mask):
        self.Hist = cv2.calcHist([image], self.Channels, mask, self.histSize, self.Ranges, self.Hist, True)

    def getDiplayHist(self):
        rawHist = cv2.normalize(self.Hist)
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

class LabHistogram(Histogram):

    def __init__(self, histResolution, scaleDisplay):
        self.histResolution = histResolution
        self.scaleDisplay = scaleDisplay
        self.histSize = [histResolution, histResolution]
        self.Ranges = [0, 255, 0, 255]
        self.Channels = [1, 2]
        self.Hist = np.zeros((histResolution, histResolution))


class RGBHistogram(Histogram):

    def __init__(self, histResolution, scaleDisplay):
        self.histResolution = histResolution
        self.scaleDisplay = scaleDisplay
        self.histSize = [histResolution, histResolution]
        self.Ranges = [0, 255, 0, 255]
        self.Channels = [0, 1]
        self.Hist = np.zeros((histResolution, histResolution))


class HSVHistogram(Histogram):

    def __init__(self, histResolution, scaleDisplay):
        self.histResolution = histResolution
        self.scaleDisplay = scaleDisplay
        self.histSize = [histResolution, histResolution]
        self.Ranges = [0, 255, 0, 255]
        self.Channels = [1, 2]
        self.Hist = np.zeros((histResolution, histResolution))




#########################
# Comparaison classes
#########################


class NaiveComp:

    def __init__(self, mode):
        self.mode = mode
        if mode == "Lab":
            self.peauHistogram = LabHistogram(32, 10)
            self.nonPeauHistogram = LabHistogram(32, 10)
        elif mode == "RGB":
            self.peauHistogram = RGBHistogram(32, 10)
            self.nonPeauHistogram = RGBHistogram(32, 10)
        elif mode == "HSV":
            self.peauHistogram = HSVHistogram(32, 10)
            self.nonPeauHistogram = HSVHistogram(32, 10)

    def addNonPeau(self, image, mask):
        self.nonPeauHistogram.addImageToHist(image, mask)

    def addPeau(self, image, mask):
        self.peauHistogram.addImageToHist(image, mask)    

    def detectPeau(self, image):

        detection = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)

        peau = cv2.normalize(self.peauHistogram.Hist)
        nonPeau = cv2.normalize(self.nonPeauHistogram.Hist)
        
        for x in range(0, image.shape[0]):
            for y in range(0, image.shape[1]):
                if self.mode == "Lab":
                    a = image.item(x, y, 1) * 31 / 255
                    b = image.item(x, y, 2) * 31 / 255
                elif self.mode == "RGB":
                    a = image.item(x, y, 0) * 31 / 255
                    b = image.item(x, y, 1) * 31 / 255
                elif self.mode == "HSV":
                    a = image.item(x, y, 0) * 31 / 255
                    b = image.item(x, y, 1) * 31 / 255
                
                if peau.item(a, b) > nonPeau.item(a, b):
                    detection.itemset((x, y, 0), 255)
                else:
                    detection.itemset((x, y, 0), 0)
        return detection

    def bayesDetectPeau(self, image):
        detection = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)

        pPeau = np.sum(self.peauHistogram.Hist)
        pNonPeau = np.sum(self.nonPeauHistogram.Hist)

        peau = cv2.normalize(self.peauHistogram.Hist)
        nonPeau = cv2.normalize(self.nonPeauHistogram.Hist)

        for x in range(0, image.shape[0]):
            for y in range(0, image.shape[1]):
                if self.mode == "Lab":
                    a = image.item(x, y, 1) * 31 / 255
                    b = image.item(x, y, 2) * 31 / 255
                elif self.mode == "RGB":
                    a = image.item(x, y, 0) * 31 / 255
                    b = image.item(x, y, 1) * 31 / 255
                elif self.mode == "HSV":
                    a = image.item(x, y, 0) * 31 / 255
                    b = image.item(x, y, 1) * 31 / 255
                if (peau.item(a, b)*pPeau + nonPeau.item(a, b)*pNonPeau) != 0:
                    detection.itemset(
                        (x, y, 0), 
                        255.0*peau.item(a, b)*pPeau/(peau.item(a, b)*pPeau + nonPeau.item(a, b)*pNonPeau)
                        )
        return detection

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


###############################
#  MAIN
###############################

with launch_ipdb_on_exception():
    #
    # First, we scan the dirs to find the test images
    #

    files = []
    dirRawPath = './Raw/'
    dirFacesPath = './Faces/'

    for dirname, dirnames, filenames in os.walk(dirRawPath):
        for filename in filenames:
            files.append(filename.strip())


    #
    # We build the histogramms for both kind of images
    #

    labComp = NaiveComp("Lab")
    rgbComp = NaiveComp("RGB")
    hsvComp = NaiveComp("HSV")

    files = files[:5]
    for f in files:
        print "Adding file : %s " % dirFacesPath + f.split('.')[0] + '_R.jpg'
        print "Adding file : %s " % dirRawPath + f

        rawImage = cv2.imread(dirRawPath + f)
        faceImage = cv2.imread(dirFacesPath + f.split('.')[0] + '_R.jpg')

        mask, antimask = generateMask(faceImage)

        
        labComp.addNonPeau(
            cv2.cvtColor(
                rawImage,
                cv2.COLOR_BGR2LAB),
            antimask
            )

        labComp.addPeau(
            cv2.cvtColor(
                faceImage,
                cv2.COLOR_BGR2LAB),
            mask
            )


        rgbComp.addNonPeau(
            cv2.cvtColor(
                rawImage,
                cv2.COLOR_BGR2RGB),
            antimask
            )

        rgbComp.addPeau(
            cv2.cvtColor(
                faceImage,
                cv2.COLOR_BGR2RGB),
            mask
            )


        hsvComp.addNonPeau(
            cv2.cvtColor(
                rawImage,
                cv2.COLOR_BGR2HSV),
            antimask
            )

        hsvComp.addPeau(
            cv2.cvtColor(
                faceImage,
                cv2.COLOR_BGR2HSV),
            mask
            )


    cv2.namedWindow("Detection Lab")
    cv2.namedWindow("Detection RGB")
    cv2.namedWindow("Detection HSV")
    cv2.namedWindow("Original")

    cv2.imshow("Original", labComp.peauHistogram.getDiplayHist())
    cv2.imshow("Detection Lab", labComp.nonPeauHistogram.getDiplayHist())
    cv2.imshow("Detection RGB", rgbComp.peauHistogram.getDiplayHist())
    cv2.imshow("Detection HSV", rgbComp.nonPeauHistogram.getDiplayHist())
    cv2.waitKey(0)

    imageToDetect = cv2.imread("./Raw/1Person0003.jpg")

    cv2.imshow("Original", imageToDetect)

    cv2.imshow("Detection Lab", labComp.bayesDetectPeau(
        cv2.cvtColor(
                imageToDetect,
                cv2.COLOR_BGR2LAB)
        ))

    cv2.imshow("Detection RGB", rgbComp.bayesDetectPeau(
        cv2.cvtColor(
                imageToDetect,
                cv2.COLOR_BGR2RGB)
        ))

    cv2.imshow("Detection HSV", hsvComp.bayesDetectPeau(
        cv2.cvtColor(
                imageToDetect,
                cv2.COLOR_BGR2HSV)
        ))

    cv2.waitKey(0)
    imageToDetect = cv2.imread("./Test/test1.jpg")

    cv2.imshow("Original", imageToDetect)

    cv2.imshow("Detection Lab", labComp.bayesDetectPeau(
        cv2.cvtColor(
                imageToDetect,
                cv2.COLOR_BGR2LAB)
        ))

    cv2.imshow("Detection RGB", rgbComp.bayesDetectPeau(
        cv2.cvtColor(
                imageToDetect,
                cv2.COLOR_BGR2RGB)
        ))

    cv2.imshow("Detection HSV", hsvComp.bayesDetectPeau(
        cv2.cvtColor(
                imageToDetect,
                cv2.COLOR_BGR2HSV)
        ))

    cv2.waitKey(0)
    imageToDetect = cv2.imread("./Test/test2.jpg")

    cv2.imshow("Original", imageToDetect)

    cv2.imshow("Detection Lab", labComp.bayesDetectPeau(
        cv2.cvtColor(
                imageToDetect,
                cv2.COLOR_BGR2LAB)
        ))

    cv2.imshow("Detection RGB", rgbComp.bayesDetectPeau(
        cv2.cvtColor(
                imageToDetect,
                cv2.COLOR_BGR2RGB)
        ))

    cv2.imshow("Detection HSV", hsvComp.bayesDetectPeau(
        cv2.cvtColor(
                imageToDetect,
                cv2.COLOR_BGR2HSV)
        ))

    cv2.waitKey(0)
