import cv2
import numpy as np
from predict import predict
import time

# Esc キー
ESC_KEY = 0x1b
# s キー
S_KEY = 0x73
MAX_FEATURE_NUM = 4
# インターバル （1000 / フレームレート）
INTERVAL = 30

class Motion:
    # コンストラクタ
    def __init__(self):
        # 表示ウィンドウ
        cv2.namedWindow("motion")
        cv2.namedWindow("canny_edges")
        # マウスイベントのコールバック登録
        cv2.setMouseCallback("motion", self.onMouse)
        # 映像
        self.video = cv2.VideoCapture(0)
        self.video.set(5,60)
        self.interval = INTERVAL
        self.frame = None
        self.gray_next = None
        self.gray_prev = None
        self.features = None
        self.frames = 1

    # メインループ
    def run(self):

        # 最初のフレームの処理
        end_flag, self.frame = self.video.read()
        output = self.frame

        while end_flag:

            self.gray_next = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            output = self.gray_next

            if self.features is not None:
                for feature in self.features:
                    cv2.circle(self.frame, (feature[0], feature[1]), 2, (0, 0, 255), -1, 8, 0)
                if len(self.features) == MAX_FEATURE_NUM:
                    pts1 = self.features
                    cv2.polylines(self.frame, [pts1.astype(np.int32)], True, (0, 0, 255), 2)
                    if pts1[0][0] - pts1[1][0] > pts1[3][0] - pts1[2][0]:
                        width = pts1[0][0] - pts1[1][0]
                    else:
                        width = pts1[3][0] - pts1[2][0]
                    if pts1[3][1] - pts1[0][1] > pts1[2][1] - pts1[1][1]:
                        height = pts1[3][1] - pts1[0][1]
                    else:
                        height = pts1[2][1] - pts1[1][1]
                    pts2 = np.float32([[width,0],[0,0],[0,height],[width,height]])
                    M = cv2.getPerspectiveTransform(pts1,pts2)
                    output = cv2.warpPerspective(self.gray_next,M,(width,height))
                    if self.frames == 60:
                        predict(output)

            canny_edges = cv2.Canny(output,100,200)

            # 表示
            cv2.imshow("canny_edges",canny_edges)
            cv2.imshow("motion", self.frame)

            # 次のループ処理の準備
            self.gray_prev = self.gray_next
            end_flag, self.frame = self.video.read()
            if end_flag:
                self.gray_next = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            self.frames += 1
            if self.frames == 61:
                self.frames = 1

            # インターバル
            key = cv2.waitKey(self.interval)
            # "Esc"キー押下で終了
            if key == ESC_KEY:
                break
            # "s"キー押下で一時停止
            elif key == S_KEY:
                self.interval = 0


        # 終了処理
        cv2.destroyAllWindows()
        self.video.release()


    def onMouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.features is None:
            self.addFeature(x, y)
            return
        else:
            self.addFeature(x, y)

        return


    def addFeature(self, x, y):
        if self.features is None:
            self.features = np.empty((0,2), float)
            self.features = np.append(self.features, np.array([[x, y]]), axis = 0).astype(np.float32)

        elif len(self.features) >= MAX_FEATURE_NUM:
            print("max feature num over: " + str(MAX_FEATURE_NUM))
            self.features = np.empty((0,2), float)
            self.features = np.append(self.features, np.array([[x, y]]), axis = 0).astype(np.float32)

        else:
            self.features = np.append(self.features, np.array([[x, y]]), axis = 0).astype(np.float32)



if __name__ == '__main__':
    Motion().run()
