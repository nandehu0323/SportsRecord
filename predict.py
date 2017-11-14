# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os

def predict(file):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
    	help="path to input image to be OCR'd")
    ap.add_argument("-p", "--preprocess", type=str, default="thresh",
    	help="type of preprocessing to be done")
    args = vars(ap.parse_args())

    # check to see if we should apply thresholding to preprocess the
    # image
    if args["preprocess"] == "thresh":
    	gray = cv2.threshold(file, 0, 255,
    		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # make a check to see if median blurring should be done to remove
    # noise
    elif args["preprocess"] == "blur":
    	gray = cv2.medianBlur(file, 3)

    # write the grayscale image to disk as a temporary file so we can
    # apply OCR to it
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)

    # load the image as a PIL/Pillow image, apply OCR, and then delete
    # the temporary file
    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    print(text)
