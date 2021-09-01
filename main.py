import tensorflow as tf
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import pandas as pd
import shutil
import glob
import random
import os
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import SGD
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files

def main():
    find_activated_gpu() # TensorflowののインポートとGPUの確認

    ### データセットの用意 ###
    id = 'id_for_dogs_vs_cats' # Google Drive上にアップロードしたdogs-vs-cats.zipの共有リンク内に含まれるid
    load_dataset(id) # データセットの読み込み
    df = pd.DataFrame()
    df = prepare_dataframe()
    split_images(df) # データセットを訓練用と検証用に分割
    confirm_splitting() # 分割したことを確認
    ### データセットの用意 ###

    ### データセットの前処理 ###
    train_data_dir = './train/train' # 訓練データ用ディレクトリのパス
    val_data_dir = './train/val' # 検証データ用ディレクトリのパス
    batch_size = 128 # 所望のバッチサイズを指定
    train_generator, val_generator = prepare_image_generators(train_data_dir, val_data_dir, batch_size) # 訓練用と検証用のミニバッチ生成インスタンス
    ### データセットの前処理 ###

    ### モデルの定義から学習結果の確認まで ###
    model = define_model() # モデルの定義
    model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=1e-3, momentum=0.9),metrics=['accuracy']) # モデルのコンパイル
    history = model.fit(train_generator,epochs=10,validation_data=val_generator,verbose=1,shuffle=True) # 学習
    check_result(history, save_fig=False) # 学習結果の確認
    ### モデルの定義から学習結果の確認まで ###

def find_activated_gpu():
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

def load_dataset(id_for_dogs_vs_cats):
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    downloaded = drive.CreateFile({'id': id_for_dogs_vs_cats})
    downloaded.GetContentFile('dogs-vs-cats.zip')

    !unzip dogs-vs-cats.zip
    !rm dogs-vs-cats.zip

    !unzip train.zip
    !rm train.zip

    !unzip test1.zip
    !rm test1.zip

def prepare_dataframe():
    df = pd.read_csv('sampleSubmission.csv')

    # ラベルの内訳は，cat = 0, dog = 0
    df = df.rename(columns={'id': 'name'})
    df = df.astype(str)
    for i in range(len(df)):
        df.iat[i,0] = 'cat.' + str(df.iat[i,0]) + '.jpg' # 画像名に拡張子を付加
        df.iat[i,0] = str(df.iat[i,0]) # 数値ではなく文字列に変換

    length = len(df)
    # 犬のindexを12500件新規作成
    for i in range(length):
        name = 'dog.'+str(i+1)+'.jpg'
        label = '1'
        index = {'name': name, 'label': label}
        df = df.append(index, ignore_index=True)
    return df

def split_images(df):
    # 訓練データ用ディレクトリと検証データ用ディレクトリの作成
    if not os.path.isdir('train/train'):
        os.mkdir('train/train')
    if not os.path.isdir('train/val'):
        os.mkdir('train/val')

    train_images = glob.glob("train/*.jpg") # imagesディレクトリの画像ファイル名を全取得

    # 7:3の割合で訓練データと検証データを分割
    each_cnt = {'0': 0, '1': 0}
    TRAIN = len(train_images)*7/10
    while not (each_cnt['0'] == TRAIN and each_cnt['1'] == TRAIN):
        for i in range(len(train_images)):
            # カウントが共に上限に達した場合，ループから抜ける
            if each_cnt['0'] == TRAIN and each_cnt['1'] == TRAIN:
                break
            image_name = df.iat[i,0]
            image_label = df.iat[i,1]
            # ランダムに画像を選定
            # ただし，カウントが上限に達していないことが条件
            if (int(random.random()*10)%7 == 0) and not (each_cnt[str(image_label)] == TRAIN):
                move_data_to_train(image_name)
                each_cnt[str(image_label)] += 1
    
    # 残りのデータを全てdataset/train/valに移動
    root_images = glob.glob("train/*.jpg")
    for i in range(len(root_images)):
        from_path = root_images[i] # 移動元パス
        to_path = 'train/val/' # 移動先パス
        shutil.move(from_path, to_path) # 画像の移動

# train/trainディレクトリに画像を移動
def move_data_to_train(image_name):
  from_path = 'train/'+image_name # 移動元パス
  to_path = 'train/train/' # 移動先パス
  if os.path.isfile(from_path) and not os.path.isfile(to_path+image_name):
    shutil.move(from_path, to_path) # 画像の移動

def confirm_splitting():
    num_of_train = len(glob.glob("train/train/*.jpg"))
    num_of_val = len(glob.glob("train/val/*.jpg"))
    total = num_of_train + num_of_val

    print('train images = '+str(num_of_train))
    print('val images = '+str(num_of_val))
    print('ratio = '+str(num_of_train/total)+':'+str(num_of_val/total))

def prepare_image_generators(df, train_data_dir, val_data_dir, batch_size):
    IMAGE_SIZE = (128, 128)

    # rescale=1.0/255で画素値を正規化
    train_datagen = ImageDataGenerator(rescale=1.0/255)
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    # .flow_from_dataframeを用いてデータのバッチ生成
    train_generator = train_datagen.flow_from_dataframe(
        df,
        directory=train_data_dir,
        x_col="name",
        y_col="label",
        weight_col=None,
        target_size=IMAGE_SIZE,
        color_mode="rgb",
        classes=None,
        class_mode="binary",
        batch_size=batch_size)

    val_generator = test_datagen.flow_from_dataframe(
        df,
        directory=val_data_dir,
        x_col="name",
        y_col="label",
        weight_col=None,
        target_size=IMAGE_SIZE,
        color_mode="rgb",
        classes=None,
        class_mode="binary",
        batch_size=batch_size)
    
    return train_generator, val_generator

def define_model():
    # 入力テンソルに(img_width, img_height, 3)のバッチを指定
    input_tensor = keras.Input(shape=(128, 128, 3))
    # 出力層の全結合層を除いてResNet50をインスタンス化
    ResNet50 = ResNet50(include_top=False, weights='imagenet',input_tensor=input_tensor)
    # モデルに層を重ねるためのメソッド
    top_model = keras.Sequential()
    # Denseレイヤは入力として1次元ベクトルを取るため，Flatten()で平滑化
    top_model.add(layers.Flatten(input_shape=ResNet50.output_shape[1:]))
    # 出力層をsigmoid関数に指定
    top_model.add(layers.Dense(1, activation='sigmoid'))
    # Modelを定義
    model = keras.Model(inputs=ResNet50.input, outputs=top_model(ResNet50.output))
    return model

def check_result(history, save_fig):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(hist['epoch'], hist['accuracy'],label='Training accuracy')
    plt.plot(hist['epoch'], hist['val_accuracy'],label = 'Validation accuracy')
    plt.ylim([0,1]) 
    plt.legend() # 凡例の表示
    # 画像の保存
    if save_fig:
        image_name = 'CatDog-learning-curve.png'
        plt.savefig(image_name)
        files.download(image_name)
    plt.show()

if __name__ == "__main__":
    main()