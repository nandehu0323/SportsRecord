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

def search_neighbor(p0, ps):
    L = np.array([])
    for i in range(ps.shape[0]):
        L = np.append(L,np.linalg.norm(ps[i]-p0))
        L[np.where(L==0)] = 999 #値が0の要素を無理やり書き換える
    return np.argmin(L),L[np.argmin(L)] #返り値は近傍点の(ラベル番号),(距離)

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

    time += 1
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

    stats = centroids = near_index = near_dist = []

    # ラベリングのため深度変更
    dst_frame = np.uint8(dst_frame)

    if time>=50:
        # ラベリング処理
        nlabels, labeledimg, stats, CoGs = cv2.connectedComponentsWithStats(dst_frame)
        region = np.empty((0,5),int)
        center = np.empty((0,2),int)

        for n in range(nlabels):
            x, y, width, height, size = stats[n]

            if size>30 and size<1000:
                #十分な大きさの領域のみ抽出
                region = np.vstack((region,stats[n]))
                center = np.vstack((center,CoGs[n]))

        for i in range(len(region)):
            x, y, width, height, size = region[i]
            near_index, near_dist = search_neighbor(center[i],center)
            if time == 50:
                print("label" + str(i) + "\n near_index: " + str(near_index) + " near_dist: " + str(near_dist))
            if near_dist <= 50:
                c1x = min(x        , region[near_index,0])
                c1y = min(y        , region[near_index,1])
                c2x = max(x+width  , region[near_index,0]+region[near_index,2])
                c2y = max(y+height , region[near_index,1]+region[near_index,3])
                cv2.rectangle(dst_frame, (c1x,c1y), (c2x,c2y), (255,255,255), 2)
            else:
                cv2.rectangle(dst_frame, (x-2, y-2), (x+width+2, y+height+2), (255,255,255), 2)

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
