#!/usr/local/bin/python
#! -*- coding: utf-8 -*-

from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime

#import sys
import cv2
import numpy as np
import keras
from keras.models import model_from_json
from keras.preprocessing.image import array_to_img, img_to_array, list_pictures, load_img
from keras import backend as K

IMAGE_SIZE = 64 # size
IMAGE_CHANNEL = 3 #RGB
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*IMAGE_CHANNEL

# 顔認識用のカスケード型識別器の読み込み
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def load_face_img(file, target_size):
    # 画像ファイル読み込み
    img = cv2.imread(file)
    
    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 顔領域の探索
    face = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    # 検出部分の切り出し。最初に検出した１つだけを切り出す（１画像に顔が１つの前提）
    if len(face):
        x, y, w, h = face[0]
        face_img = img[y:y+h, x:x+w]
    else:
        face_img = img
    
    # サイズの変更
    face_img = cv2.resize(face_img, target_size)
    
    return np.asarray(face_img)

app = Flask(__name__)
@app.route('/', methods = ['GET', 'POST'])
def classify():
  if request.method == 'GET':
    return render_template('index.html')
  if request.method == 'POST':
    # アプロードされたファイルを保存する
    f = request.files['file']
    filename = "./static/" + datetime.now().strftime("%Y%m%d%H%M%S") + ".jpg"
    f.save(filename)
    # モデルを使って判定する
    img = img_to_array(load_face_img(filename, target_size=(IMAGE_SIZE, IMAGE_SIZE)))
    x_val = img

    x_val = np.asarray(x_val)
    x_val = x_val.astype('float32')
    x_val = x_val / 255.0
    
    if K.image_data_format() == 'channels_first':
        x_val = x_val.reshape(1, IMAGE_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
        input_shape = (1, IMAGE_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
    else:
        x_val = x_val.reshape(1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL)
        input_shape = (1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL)
    
    #モデルを読み込む
    model = model_from_json(open('model.json').read())

    # 学習結果を読み込む
    model.load_weights('model.h5')

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    # prediction
    predict = model.predict_classes(x_val)
    proba = model.predict_proba(x_val)[0][1]*100 # probability of class 1

    return render_template('index.html', filepath = filename , predict = predict, proba = proba )
    
if __name__ == '__main__':
  app.run(host='0.0.0.0', port=int("5000"),debug=True)
