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

    def __init__(self, histResolution, scaleDisplay, mode):
        self.mode = mode
        if mode == "LAB":
            self.histResolution = histResolution
            self.scaleDisplay = scaleDisplay
            self.histSize = [histResolution, histResolution]
            self.Ranges = [0, 255, 0, 255]
            self.Channels = [1, 2]
            self.Hist = np.zeros((histResolution, histResolution))
        
        elif mode == "RGB":
            self.histResolution = histResolution
            self.scaleDisplay = scaleDisplay
            self.histSize = [histResolution, histResolution]
            self.Ranges = [0, 255, 0, 255]
            self.Channels = [0, 1]
            self.Hist = np.zeros((histResolution, histResolution))
        
        elif mode == "HSV":
            self.histResolution = histResolution
            self.scaleDisplay = scaleDisplay
            self.histSize = [histResolution, histResolution]
            self.Ranges = [0, 179, 0, 255]
            self.Channels = [0, 1]
            self.Hist = np.zeros((histResolution, histResolution))
        
        else:
            raise Exception("Unsupported mode")

    def addImageToHist(self, image, mask):
        if self.mode == "LAB":
            self.Hist = cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_BGR2LAB)], self.Channels, mask, self.histSize, self.Ranges, self.Hist, True)
        elif self.mode == "RGB":
            self.Hist = cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_BGR2RGB)], self.Channels, mask, self.histSize, self.Ranges, self.Hist, True)
        elif self.mode == "HSV":
            self.Hist = cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_BGR2HSV)], self.Channels, mask, self.histSize, self.Ranges, self.Hist, True)

    def getDiplayHist(self):
        
        maxHist = self.Hist.max()    
        normalizedHist = cv2.normalize(self.Hist)
        displayHist = np.zeros((self.histResolution * self.scaleDisplay, self.histResolution * self.scaleDisplay))
        for a in range(0, 32):
            for b in range(0, 32):
                cv2.rectangle(
                    displayHist,
                    (a*self.scaleDisplay, b*self.scaleDisplay),
                    ((a + 1)*self.scaleDisplay - 1, (b + 1)*self.scaleDisplay - 1),
                    self.Hist.item(a, b)/maxHist, 
                    #normalizedHist.item(a, b), 
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
            if math.sqrt(image.item(x, y, 0)**2 + image.item(x, y, 1)**2 + image.item(x, y, 2)**2) > math.sqrt(3*252**2):
                mask.itemset((x, y, 0), 255)
                antimask.itemset((x, y, 0), 0)
            else:
                mask.itemset((x, y, 0), 0)
                antimask.itemset((x, y, 0), 255)
    return antimask, mask

def writeHistToFile(filename, hist):
    f = open(filename, 'w')
    for x in range (0, 32):
        for y in range(0, 32):
            f.write(str(hist.item((x, y))) + ';')
    f.close()

def readHistFromFile(filename):
    hist = np.zeros((32, 32))

    f = open(filename, 'r')
    raw = f.readline()
    f.close()
    values = raw.split(';')
    index = 0
    for x in range (0, 32):
        for y in range(0, 32):
            hist.itemset((x, y), values[index])
            index += 1
    return hist



###############################
#  MAIN
###############################

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
    labHistPeau = Histogram(histRange, scale, "LAB")
    rgbHistPeau = Histogram(histRange, scale, "RGB")
    hsvHistPeau = Histogram(histRange, scale, "HSV")

    labHist = Histogram(histRange, scale, "LAB")
    rgbHist = Histogram(histRange, scale, "RGB")
    hsvHist = Histogram(histRange, scale, "HSV")

    #files = files[:10]
    for f in files:
        print "Adding file : %s " % dirFacesPath + f.split('.')[0] + '_R.jpg'
        print "Adding file : %s " % dirRawPath + f

        rawImage = cv2.imread(dirRawPath + f)
        faceImage = cv2.imread(dirFacesPath + f.split('.')[0] + '_R.jpg')

        mask, antimask = generateMask(faceImage)

        
        labHist.addImageToHist(
            rawImage,
            antimask
            )
        
        labHistPeau.addImageToHist(
            faceImage,
            mask
            )

        rgbHist.addImageToHist(
            rawImage,
            antimask
            )
        rgbHistPeau.addImageToHist(
            faceImage,
            mask
            )

        hsvHist.addImageToHist(
            rawImage,
            antimask
            )
        hsvHistPeau.addImageToHist(
            faceImage,
            mask
            )

    cv2.namedWindow("Lab Histogram")
    cv2.imshow("Lab Histogram", labHist.getDiplayHist())
    cv2.namedWindow("RGB Histogram")
    cv2.imshow("RGB Histogram", rgbHist.getDiplayHist())
    cv2.namedWindow("HSV Histogram")
    cv2.imshow("HSV Histogram", hsvHist.getDiplayHist())


    cv2.namedWindow("Lab Histogram Peau")
    cv2.imshow("Lab Histogram Peau", labHistPeau.getDiplayHist())
    cv2.namedWindow("RGB Histogram Peau")
    cv2.imshow("RGB Histogram Peau", rgbHistPeau.getDiplayHist())
    cv2.namedWindow("HSV Histogram Peau")
    cv2.imshow("HSV Histogram Peau", hsvHistPeau.getDiplayHist())

    cv2.waitKey(0)

    writeHistToFile("HSV-PEAU-HIST.hist", hsvHistPeau.Hist)
    writeHistToFile("HSV-NONPEAU-HIST.hist", hsvHist.Hist)

    writeHistToFile("RGB-PEAU-HIST.hist", rgbHistPeau.Hist)
    writeHistToFile("RGB-NONPEAU-HIST.hist", rgbHist.Hist)

    writeHistToFile("LAB-PEAU-HIST.hist", labHistPeau.Hist)
    writeHistToFile("LAB-NONPEAU-HIST.hist", labHist.Hist)

    print "Now checking saved histograms...\n"

    savedlabHistPeau = Histogram(histRange, scale, "LAB")
    savedrgbHistPeau = Histogram(histRange, scale, "RGB")
    savedhsvHistPeau = Histogram(histRange, scale, "HSV")

    savedlabHist = Histogram(histRange, scale, "LAB")
    savedrgbHist = Histogram(histRange, scale, "RGB")
    savedhsvHist = Histogram(histRange, scale, "HSV")

    savedlabHistPeau.Hist = readHistFromFile("LAB-PEAU-HIST.hist")
    savedlabHist.Hist = readHistFromFile("LAB-NONPEAU-HIST.hist")

    savedrgbHistPeau.Hist = readHistFromFile("RGB-PEAU-HIST.hist")
    savedrgbHist.Hist = readHistFromFile("RGB-NONPEAU-HIST.hist")
    
    savedhsvHistPeau.Hist = readHistFromFile("HSV-PEAU-HIST.hist")
    savedhsvHist.Hist = readHistFromFile("HSV-NONPEAU-HIST.hist")

    if np.array_equal(savedlabHistPeau.Hist, labHistPeau.Hist):
        "ERROR - labHistPeau not the same as saved instance"
    if np.array_equal(savedlabHist.Hist, labHist.Hist):
        "ERROR - labHist not the same as saved instance"
    if np.array_equal(savedrgbHistPeau.Hist, rgbHistPeau.Hist):
        "ERROR - rgbHistPeau not the same as saved instance"
    if np.array_equal(savedrgbHist.Hist, rgbHist.Hist):
        "ERROR - rgbHist not the same as saved instance"
    if np.array_equal(savedhsvHistPeau.Hist, hsvHistPeau.Hist):
        "ERROR - hsvHistPeau not the same as saved instance"
    if np.array_equal(savedhsvHist.Hist, hsvHist.Hist):
        "ERROR - hsvHist not the same as saved instance"  

    cv2.imshow("Lab Histogram", savedlabHist.getDiplayHist())
    cv2.imshow("RGB Histogram", savedrgbHist.getDiplayHist())
    cv2.imshow("HSV Histogram", savedhsvHist.getDiplayHist())

    cv2.imshow("Lab Histogram Peau", savedlabHistPeau.getDiplayHist())
    cv2.imshow("RGB Histogram Peau", savedrgbHistPeau.getDiplayHist())
    cv2.imshow("HSV Histogram Peau", savedhsvHistPeau.getDiplayHist())

    
    
    cv2.waitKey(0)

