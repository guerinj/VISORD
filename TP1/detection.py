from __future__ import division
import cv2
import numpy as np
import os
import ipdb
import math
from ipdb import launch_ipdb_on_exception
from histograms import *


#########################
# Comparaison classes
#########################


class Comp:

    def __init__(self, mode):
        self.mode = mode
        if mode == "LAB":
            self.peauHistogram = Histogram(32, 10, "LAB")
            self.nonPeauHistogram = Histogram(32, 10, "LAB")

            self.peauHistogram.Hist = readHistFromFile("LAB-PEAU-HIST.hist")
            self.nonPeauHistogram.Hist = readHistFromFile("LAB-NONPEAU-HIST.hist")
        elif mode == "RGB":
            self.peauHistogram = Histogram(32, 10, "RGB")
            self.nonPeauHistogram = Histogram(32, 10, "RGB")

            self.peauHistogram.Hist = readHistFromFile("RGB-PEAU-HIST.hist")
            self.nonPeauHistogram.Hist = readHistFromFile("RGB-NONPEAU-HIST.hist")
        elif mode == "HSV":
            self.peauHistogram = Histogram(32, 10, "HSV")
            self.nonPeauHistogram = Histogram(32, 10, "HSV")

            self.peauHistogram.Hist = readHistFromFile("HSV-PEAU-HIST.hist")
            self.nonPeauHistogram.Hist = readHistFromFile("HSV-NONPEAU-HIST.hist")
        else:
            raise Exception("Unsupported mode")
 
                

    def naiveDetectPeau(self, image):
        
        detection = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)

        peau = self.peauHistogram.Hist/self.peauHistogram.Hist.sum()
        nonPeau = self.nonPeauHistogram.Hist/self.nonPeauHistogram.Hist.sum()
        
        for x in range(0, image.shape[0]):
            for y in range(0, image.shape[1]):
                if self.mode == "LAB":
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

        sumPeau = self.peauHistogram.Hist.sum()
        sumNonPeau = self.nonPeauHistogram.Hist.sum()

        peau = self.peauHistogram.Hist/sumPeau
        nonPeau = self.nonPeauHistogram.Hist/sumNonPeau

        for x in range(0, image.shape[0]):
            for y in range(0, image.shape[1]):
                if self.mode == "LAB":
                    a = image.item(x, y, 1) * 31 / 255
                    b = image.item(x, y, 2) * 31 / 255
                elif self.mode == "RGB":
                    a = image.item(x, y, 0) * 31 / 255
                    b = image.item(x, y, 1) * 31 / 255
                elif self.mode == "HSV":
                    a = image.item(x, y, 0) * 31 / 255
                    b = image.item(x, y, 1) * 31 / 255

                if (peau.item(a, b)*sumPeau + nonPeau.item(a, b)*sumNonPeau) != 0:    
                    v = 255.0*peau.item(a, b)*sumPeau/(peau.item(a, b)*sumPeau + nonPeau.item(a, b)*sumNonPeau)                
                    detection.itemset(
                        (x, y, 0), 
                        v
                        )
        #detection = cv2.normalize(detection)
        return detection


###############################
#  MAIN
###############################

with launch_ipdb_on_exception():

    
    labComp = Comp("LAB")
    rgbComp = Comp("RGB")
    hsvComp = Comp("HSV")

    #
    # Now we try to detect skin on an image
    #

    files = []
    dirTestPath = './Test/'
    

    for dirname, dirnames, filenames in os.walk(dirTestPath):
        for filename in filenames:
            print filename
            if filename.strip() != '.DS_Store':
                files.append(filename.strip())

    for f in files :
        print "Trying to detect on : %s " % dirTestPath + f
        imageToDetect = cv2.imread(dirTestPath + f)
        
        cv2.namedWindow("Original")
        cv2.namedWindow("Detection RGB Naive")
        cv2.namedWindow("Detection RGB Bayes")
        cv2.namedWindow("Detection LAB Naive")
        cv2.namedWindow("Detection LAB Bayes")
        cv2.namedWindow("Detection HSV Naive")
        cv2.namedWindow("Detection HSV Bayes")

        cv2.imshow("Original", imageToDetect)
        cv2.imshow("Detection LAB Naive", labComp.naiveDetectPeau(imageToDetect))
        cv2.imshow("Detection RGB Naive", rgbComp.naiveDetectPeau(imageToDetect))
        cv2.imshow("Detection HSV Naive", hsvComp.naiveDetectPeau(imageToDetect))

        cv2.imshow("Detection LAB Bayes", labComp.bayesDetectPeau(imageToDetect))
        cv2.imshow("Detection RGB Bayes", rgbComp.bayesDetectPeau(imageToDetect))
        cv2.imshow("Detection HSV Bayes", hsvComp.bayesDetectPeau(imageToDetect))
        

        cv2.waitKey(0)