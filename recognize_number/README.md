# 機械学習 数字判別(Linux or Mac or Bash on Windows)

### ライブラリのインストール


```
beautifulsoup4
sklearn
h5py
keras
```




### 画像の振り分け
pngがいいなああああああ。
```
 ch7/
    ├ number_downloader.py
    ├ image
        ├ 0 --- 0の画像
        ├ 1 --- 1の画像
        ├ 2 --- 2の画像
        ├ 3 --- 3の画像
        ├ 4 --- 4の画像
        ├ 5 --- 5の画像
        ├ 6 --- 6の画像
        ├ 7 --- 7の画像
        ├ 8 --- 8の画像
        └ 9 --- 9の画像
```

## 機械学習 学習

### 画像を数値データに変換
python の Numpy を使用して振り分けた画像を元に数値データを作成する。

```
$ python number-makedata.py
--- 0 を処理中
--- 1 を処理中
--- 2 を処理中
--- 3 を処理中
・
・
・
ok, 200
```

number-makedata.py を実行すると「image/number.npy」という Numpy のデータが作成される。

### CNNで学習
Numpyのデータを畳み込みニュートラルネットワーク(CNN)で学習させる。
```
$ python number_keras.py
Using TensorFlow backend.
...
loss= 0.86482
accuracy= 0.90133
```
この場合、正解率は0.865(85%)を意味する。


## 画像判別
学習させたモデルを元に、画像を判別する。

```
$ python number-checker.py (画像パス1) (画像パス2) ...
```


## 機械学習 精度向上
正解率を上げるために、チューニングを行う。画像の角度を変えたり反転させたりしてデータ数を増やすか。

先ほど使用した number-makedata.py を改良したものが number-makedata2.py になる。コマンドから以下を実行して画像データを水増しできるー

```
$ python number-makedata2.py
ok, 2020
```

画像データが増えた状態で学習！！！

```
$ python number_keras2.py
```


画像判別は、前回同様のコマンドを実行！

```
$ python number-checker.py (画像パス1) (画像パス2) ...
```

