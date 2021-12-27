# CatDog-Classification
## 概要
CNNを用いた分類モデルの構築を練習するために犬猫分類モデルを構築しました。

## 実装
TensorFlowをバックエンドとしてKerasを用いて実装しました。

## データセット
kaggleにある以下のデータセットを利用しました。
* https://www.kaggle.com/c/dogs-vs-cats/overview

## ネットワークアーキテクチャ
ResNet50を特徴抽出器として、ResNet50の元の出力層から2値分類用の全結合層に取り替えた。

## 学習結果
![CatDog-learning-curve (2)](https://user-images.githubusercontent.com/49334354/130533772-89ade01c-d714-437c-84e6-5f08eb10010f.png)
