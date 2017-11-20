# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os

def predict(file,preprocess="thresh"):
    # check to see if we should apply thresholding to preprocess the
    # image
    if preprocess == "thresh":
    	gray = cv2.threshold(file, 0, 255,
    		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # make a check to see if median blurring should be done to remove
    # noise
    elif preprocess == "blur":
    	gray = cv2.medianBlur(file, 3)

    # write the grayscale image to disk as a temporary file so we can
    # apply OCR to it
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)

    # load the image as a PIL/Pillow image, apply OCR, and then delete
    # the temporary file
    text = pytesseract.image_to_string(Image.open(filename), config='--psm 7 --eom 3 nobatch digits', lang="num")
    os.remove(filename)
    return text
