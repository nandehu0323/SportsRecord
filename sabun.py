import cv2
import numpy as np

ESC_KEY = 27
INTERVAL= 33
FRAME_RATE = 60

WINDOW_SRC = "src"
WINDOW_DIFF = "diff"
WINDOW_WB = "WhiteBlack"
WINDOW_CUTBK = "Cut background"

FILE_ORG = "ball2-1.mov"

#p0に最も近い点を求める関数search_neighbor
def search_neighbor(p0,ps):
    L = np.array([])
    for i in range(ps.shape[0]):
        L = np.append(L,np.linalg.norm(ps[i])-p0)
    return np.argmin(L),min(L) #返り値は近傍点の(ラベル番号),(距離)

#ウィンドウ命名
cv2.namedWindow(WINDOW_SRC)
cv2.namedWindow(WINDOW_DIFF)
cv2.namedWindow(WINDOW_WB)

# 元d
mov_org = cv2.VideoCapture(FILE_ORG)

# 最初のフレーム読み込み
has_next, i_frame = mov_org.read()

# 背景フレーム
back_frame = np.zeros_like(i_frame, np.float32)

time = 0 #経過時間(フレーム)
while has_next == True:

    time +=1
    # print(time)
    # 入力画像を浮動小数点型に変換
    f_frame = i_frame.astype(np.float32)

    # 差分
    diff_frame = cv2.absdiff(f_frame, back_frame)

    # 2値化
    gray_frame = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)
    thresh = 100 #しきい値
    max_pixel = 255
    ret, dst_frame = cv2.threshold(gray_frame,
                                thresh,
                                max_pixel,
                                cv2.THRESH_BINARY)

    stats = centroids = near_index = near_dis = []

    # ラベリングのため深度変更
    dst_frame = np.uint8(dst_frame)

    if time>=50:
        # ラベリング処理
        nlabels, labeledimg, stats, CoGs = cv2.connectedComponentsWithStats(dst_frame)


        for n in range(nlabels):
            x,y,width,height,size = stats[n]
            near_index, near_dis = search_neighbor(CoGs[n],CoGs)
            if size<30:
                nlabels -= 1
                stats.delete()
                CoGs.delete()
            else:
                #cv2.rectangle(dst_frame, (x-2, y-2), (x+width+2, y+height+2), (255,255,255), 2)
                cv2.rectangle(diff_frame, (x-2, y-2), (x+width+2, y+height+2), (255,255,255), 2)

                if time == 70:
                #print("[time=70,point" + str(n) + "]\n)
                    print("[time=70,point" + str(n) + "]\n near_index: " + str(near_index) + " near_dis: " + str(near_index) )


    # 背景の更新
    cv2.accumulateWeighted(f_frame, back_frame, 0.025)

    # フレーム表示
    cv2.imshow(WINDOW_SRC, i_frame)
    cv2.imshow(WINDOW_DIFF, diff_frame.astype(np.uint8))
    cv2.imshow(WINDOW_WB, dst_frame)

    # Escキーで終了
    key = cv2.waitKey(INTERVAL)
    if key == ESC_KEY:
        break

    # 次のフレーム読み込み
    has_next, i_frame = mov_org.read()

# 終了処理
cv2.destroyAllWindows()
mov_org.release()
