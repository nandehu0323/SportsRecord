import number_keras as number
import sys, os
from PIL import Image
import numpy as np

# コマンドラインからファイル名を得る --- (※1)
if len(sys.argv) <= 1:
    print("number-checker.py (ファイル名)")
    quit()

image_size = 50
categories = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# 入力画像をNumpyに変換 --- (※2)
X = []
files = []
for fname in sys.argv[1:]:
    img = Image.open(fname)
    img = img.convert("RGB")
    img = img.resize((image_size, image_size))
    in_data = np.asarray(img)
    X.append(in_data)
    files.append(fname)
X = np.array(X)

# CNNのモデルを構築 --- (※3)
model = number.build_model(X.shape[1:])
model.load_weights("./image/number-model.hdf5")

# データを予測 --- (※4)
html = ""
pre = model.predict(X)
for i, p in enumerate(pre):
    y = p.argmax()
    print("+ 入力:", files[i])
    print("| 得点:", categories[y])
