
import cv2
import pytesseract
import numpy as np

cv2.namedWindow("#iothack15")
cap = cv2.VideoCapture(0)


# Capture frames from the camera
while True:
    ret, output = cap.read()
    if not ret:
        continue

    output2 = cv2.rectangle(output,(350,200),(800,500),(0,0,255),3)
    output3 = output[200:500,350:800]
    #Gray Scaleで動画を読み込み
    gray = cv2.cvtColor(output3, cv2.COLOR_BGR2GRAY, dstCn=0)
    #Cannyアルゴリズムでエッジ検出
    canny_edges = cv2.Canny(gray,100,200)

    # ソース
    cv2.imshow('source', output2)
    #結果表示
    cv2.imshow('canny_edges',canny_edges)

    # clear stream for next frame
    #rawCapture.truncate(0)

    # Wait for the magic key
    keypress = cv2.waitKey(1) & 0xFF
    if keypress == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
