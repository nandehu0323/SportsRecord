# -*- coding: utf-8 -*-
import cv2
import sys
import numpy as np

VIDEO_DATA = "ball2-1.mov"

# フレーム差分の計算
def frame_sub(src1, src2, src3, th):
    # フレームの絶対差分
    d1 = cv2.absdiff(src1, src2)
    d2 = cv2.absdiff(src2, src3)

    # 二値化処理
    d1[d1 < th] = 0
    d1[d1 >= th] = 255

    # 二値化処理
    d2[d2 < th] = 0
    d2[d2 >= th] = 255

    # 元画像を2値化
    src2[src2 < th] = 0
    src2[src2 >= th] = 255
    # 2つの差分画像の論理積
    diff = cv2.bitwise_and(d1, d2)

    # ゴマ塩ノイズ除去
    mask = cv2.medianBlur(diff, 3)

    # 色を反転
    mask = cv2.bitwise_not(mask)

    return  mask

def main():
    # カメラのキャプチャ
    cap = cv2.VideoCapture(VIDEO_DATA)

    # フレームを3枚取得してグレースケール変換
    frame1 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
    frame2 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
    frame3 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)

    src = cap.read()[1]

    while(cap.isOpened()):
        # フレーム間差分を計算
        mask = frame_sub(frame1, frame2, frame3, th=10)

        # 元の画像からmaskの部分のみ表示
        src[mask == 255] = 255

        # 人検出
        # hog = cv2.HOGDescriptor()
        # hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        # hogParams = {'winStride': (8, 8), 'padding': (32, 32), 'scale': 1.05}
        # human, r = hog.detectMultiScale(mask, **hogParams)
        # for(x, y, w, h) in human:
        #     cv2.rectangle(src, (x, y), (x+w, y+h), (255, 255, 255), -1)
        # cv2.imshow("human detection", src)

        # 結果を表示
        # cv2.imshow("Frame2", frame2)
        cv2.imshow("Mask", src)

        # 3枚のフレームを更新
        frame1 = frame2
        frame2 = frame3
        frame3 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)

        src = cap.read()[1]
        # qキーが押されたら途中終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
