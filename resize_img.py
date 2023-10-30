### サイズばらばら、縦横比まちまちの画像(3チャンネル)群 -> サイズ統一(128x128)、パディングして正方形化、白黒化
### ファイル名=日付 -> 0からナンバリング

import csv
import cv2
import math
import time

list_path = ["./COVID-19/COVID-19/train/COVID/COVID_list.txt", "./COVID-19/COVID-19/train/NonCOVID/NonCOVID_list.txt"]
train_img_list_COVID = []
train_img_list_NonCOVID = []
with open(list_path[0]) as f:
    reader = csv.reader(f)
    for row in reader:
        train_img_list_COVID.append("./COVID-19/COVID-19/train/COVID/" + row[0])
with open(list_path[1]) as f:
    reader = csv.reader(f)
    for row in reader:
        train_img_list_NonCOVID.append("./COVID-19/COVID-19/train/NonCOVID/" + row[0])


n = len(train_img_list_COVID) # COVID画像数
m = len(train_img_list_NonCOVID) # NonCOVID画像数

# COVID画像
i = 0
start = time.time()
while i < n:
    img = cv2.imread(train_img_list_COVID[i])

    height, width, color = img.shape # 画像の縦横サイズを取得

    diffsize = width - height
    
    padding_half = int(diffsize / 2)
    if diffsize % 2 == 0:
        padding_img = cv2.copyMakeBorder(img, padding_half, padding_half, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    else:
        padding_img = cv2.copyMakeBorder(img, padding_half+1, padding_half, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    padding_img = cv2.resize(padding_img, (128, 128), cv2.INTER_NEAREST) # 128x128にリサイズ
    padding_img = cv2.cvtColor(padding_img, cv2.COLOR_BGR2GRAY) # チャンネル数3->1
    cv2.imwrite('./COVID-19_pad/train/COVID/COVID_{}.png'.format(i), padding_img)
    
    if i % 10 == 0:
        end = time.time()
        print("COVID画像", i+1, "/", n, "：", round(end-start, 3), "秒経過")
    i += 1

print("completed COVID画像!!")

# NonCOVID画像
i = 0
start = time.time()
while i < m:
    img = cv2.imread(train_img_list_NonCOVID[i])

    height, width, color = img.shape # 画像の縦横サイズを取得

    diffsize = width - height
    padding_half = int(diffsize / 2)
    if diffsize % 2 == 0:
        padding_img = cv2.copyMakeBorder(img, padding_half, padding_half, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    else:
        padding_img = cv2.copyMakeBorder(img, padding_half+1, padding_half, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    padding_img = cv2.resize(padding_img, (128, 128), cv2.INTER_NEAREST) # 128x128にリサイズ
    padding_img = cv2.cvtColor(padding_img, cv2.COLOR_BGR2GRAY) # チャンネル数3->1
    cv2.imwrite('./COVID-19_pad/train/NonCOVID/train/NonCOVID_{}.png'.format(i), padding_img)
    
    if i % 10 == 0:
        end = time.time()
        print("NonCOVID画像", i+1, "/", m, "：", round(end-start, 3), "秒経過")
    i += 1
print("completed NonCOVID画像!!")
