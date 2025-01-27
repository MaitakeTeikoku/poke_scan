# https://www.tensorflow.org/tutorials/images/classification?hl=ja

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import tensorflowjs as tfjs

from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import pathlib

epochs = 15

drive_dir = '/content/drive/MyDrive/development/poke_scan'
images_dir = f'/content/poke_scan/images'

# 画像数をカウント
data_dir = pathlib.Path(images_dir).with_suffix('')
image_count = len(list(data_dir.glob('*/*.png')))
print(f"画像数: {image_count}")

# 画像の読み込み
batch_size = 32
img_height = 224
img_width = 224

# 訓練データセットを作成
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# 検証データセットを作成
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# クラス名を取得（ディレクトリ名のアルファベット順）
class_names = train_ds.class_names
print(f"クラス数: {len(class_names)}")

# データ拡張
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1),
  ]
)
# データ拡張をトレーニングデータに適用
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# キャッシュとプリフェッチをして、パフォーマンスを改善
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# モデルを作成
num_classes = len(class_names)

model = Sequential([
    # 入力の正規化
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    # 畳み込み層ブロック1
    layers.Conv2D(32, 5, padding='same', activation='relu'),
    layers.MaxPooling2D(),   
    # 畳み込み層ブロック2
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    # 畳み込み層ブロック3
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    # 畳み込み層ブロック4
    layers.Conv2D(256, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    # ドロップアウト
    layers.Dropout(0.2),
    # 全結合層
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(num_classes, activation='softmax', name="outputs")
])

# モデルをコンパイル
#optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# モデルの概要を表示
model.summary()

# モデルをトレーニング、早期終了を設定
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  callbacks=[early_stopping]
)

# 早期終了のエポック数を表示
file_name_epochs = epochs
if early_stopping.stopped_epoch > 0:
  file_name_epochs = f"{early_stopping.stopped_epoch + 1}-{epochs}"
  epochs = early_stopping.stopped_epoch + 1
  print(f"早期終了 エポック数: {early_stopping.stopped_epoch + 1}")
else:
  print("早期終了なし")

# ファイル名
japan_tz = pytz.timezone("Asia/Tokyo")
now = datetime.now(japan_tz)
now = now.strftime("%Y%m%d-%H%M%S")
file_name = f"{now}_{num_classes}classes_{file_name_epochs}epochs"

# トレーニングの結果をグラフ化
try:
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs_range = range(epochs)

  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')

  # グラフで視覚化したトレーニングの結果を画像で保存
  results_dir = f'{drive_dir}/results'
  pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)

  result_name = f"{results_dir}/{file_name}.png"
  plt.savefig(result_name)

  plt.show()
except Exception as e:
  print(f"グラフ エラー: {e}")

# モデルを保存するディレクトリを指定
models_dir = f'{drive_dir}/models'
pathlib.Path(models_dir).mkdir(parents=True, exist_ok=True)

# Keras、savedmodel形式で保存
# https://www.tensorflow.org/guide/keras/save_and_serialize?hl=ja
try:
  model.save(f'{models_dir}/{file_name}.keras')
  tf.saved_model.save(model, f'{models_dir}/{file_name}_savedmodel')
except Exception as e:
  print(f"keras エラー: {e}")

# TensorFlow.js モデルに変換して保存
# https://www.tensorflow.org/js/tutorials/conversion/import_keras?hl=ja
# !pip install tensorflowjs
try:
  tfjs.converters.save_keras_model(model, f'{models_dir}/{file_name}_tfjs')
except Exception as e:
  print(f"tfjs エラー: {e}")

# TensorFlow Lite モデルに変換して保存
try:
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()

  with open(f'{models_dir}/{file_name}.tflite', 'wb') as f:
    f.write(tflite_model)
except Exception as e:
  print(f"tflite エラー: {e}")
