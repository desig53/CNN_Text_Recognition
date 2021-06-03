# CNN_Text_Recognition
Text recognition by CNN model (train&amp;test)
文字辨識練習，使用CNN來做訓練與預測，目前辨識數字0-9

files:

model.h5 : 用於預測文字的model，cnn_redo.py將會訓練此model，並用於圖片上的辨識

images : 用來預測的圖片

cnn_redo.py : 

程式功能，包含: 

1.資料處理

2.訓練model

3.預測圖片

相關套件與資料集:

Python版本 3.7.3

辨識資料集為 : keras datasets 的 mnist

keras 2.3.1

pandas 1.1.3

matplotlib 3.3.3

PIL 7.1.1
