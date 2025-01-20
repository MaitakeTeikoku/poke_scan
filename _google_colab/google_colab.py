"""
https://www.tensorflow.org/tutorials/images/classification?hl=ja
1. セットアップ
"""
# TensorFlow とその他の必要なライブラリをインポートします。
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

# 追加のライブラリ
import tensorflowjs as tfjs

epochs = 15

drive_dir = '/content/drive/MyDrive/development/poke_scan'

"""
2. データセットをダウンロードして調査する
"""
# 使用するデータセットを指定します。
images_dir = f'/content/poke_scan/images'
data_dir = pathlib.Path(images_dir).with_suffix('')

# PNGファイルの数を数えてみます。
image_count = len(list(data_dir.glob('*/*.png')))
print(f"画像数: {image_count}")

"""
3. Keras ユーティリティを使用してデータを読み込む
3-1. データセットを作成する
"""
# ローダーのいくつかのパラメーターを定義します。
batch_size = 32
img_height = 180
img_width = 180

# モデルを開発するときは、検証分割を使用することをお勧めします。
# ここでは、画像の 80％ をトレーニングに使用し、20％ を検証に使用します。
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# クラス名は、これらのデータセットのclass_names属性にあります。 
# これらはアルファベット順にディレクトリ名に対応します。
class_names = train_ds.class_names
print(f"クラス名: {class_names}")

"""
5. データセットを構成してパフォーマンスを改善する
"""
# I/O がブロックされることなくディスクからデータを取得できるように、必ずバッファ付きプリフェッチを使用します。
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

"""
6. データを標準化する
"""
# RGB チャネル値は [0, 255] の範囲にあり、ニューラルネットワークには理想的ではありません。
# 一般に、入力値は小さくする必要があります。
normalization_layer = layers.Rescaling(1./255)

# Dataset.map を呼び出すことにより、データセットに適用できます。
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

"""
7. 基本的な Keras モデル
7-1. モデルを作成する
"""
# Sequential モデルは、それぞれに最大プールレイヤー （tf.keras.layers.MaxPooling2D）を持つ 3 つの畳み込みブロック（tf.keras.layers.Conv2D）で構成されます。
# ReLU 活性化関数（'relu'）により活性化されたユニットが 128 個ある完全に接続されたレイヤー （tf.keras.layers.Dense）があります。
# このチュートリアルの目的は、標準的なアプローチを示すことなので、このモデルは高精度に調整されていません。
num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2), # ドロップアウト
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, name="outputs")
])

"""
7-2. モデルをコンパイルする
"""
# このチュートリアルでは、tf.keras.optimizers.Adam オプティマイザとtf.keras.losses.SparseCategoricalCrossentropy 損失関数を選択します。
# 各トレーニングエポックのトレーニングと検証の精度を表示するには、Model.compile に metrics 引数を渡します。
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

"""
7-3. モデルの概要
"""
# Keras の Model.summary メソッドを使用して、ネットワークのすべてのレイヤーを表示します。
model.summary()

"""
7-4. モデルをトレーニングする
"""
# Keras Model.fit メソッドを使用して、10 エポックのモデルをトレーニングします。
#epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

"""
8. トレーニングの結果を視覚化する
"""
# トレーニングセットと検証セットで損失と精度のプロットを作成します。
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
#plt.show()

"""
グラフで視覚化したトレーニングの結果を画像で保存
"""
file_name = f"{num_classes}_{epochs}epochs"
results_dir = f'{drive_dir}/results'
pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)

result_name = f"{results_dir}/result_{file_name}.png"
plt.savefig(result_name)

"""
Keras モデルを保存する
https://www.tensorflow.org/guide/keras/save_and_serialize?hl=ja
"""
# モデルを保存する
models_dir = f'{drive_dir}/models'
pathlib.Path(models_dir).mkdir(parents=True, exist_ok=True)
try:
  model.save(f'{models_dir}/model_{file_name}.keras')
except Exception as e:
  print(f"keras エラー: {e}")

"""
TensorFlow.js モデルを保存する
https://www.tensorflow.org/js/tutorials/conversion/import_keras?hl=ja
!pip install tensorflowjs
"""
try:
# TensorFlow.jsモデルとして保存
  tfjs.converters.save_keras_model(model, f'{models_dir}/model_{file_name}_tfjs')
except Exception as e:
  print(f"tfjs エラー: {e}")

"""
12. TensorFlow Lite を使用する
12-1. Keras Sequential モデルを TensorFlow Lite モデルに変換する
"""
# トレーニング済みの Keras Sequential モデルを取得し、tf.lite.TFLiteConverter.from_keras_model を使用して TensorFlow Lite モデルを生成します。
# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open(f'{models_dir}/model_{file_name}.tflite', 'wb') as f:
  f.write(tflite_model)

"""
11. 新しいデータを予測する
"""
# モデルを使用して、トレーニングセットまたは検証セットに含まれていなかった画像を分類します。
new_data_url = "https://zukan.pokemon.co.jp/zukan-api/up/images/index/7b705082db2e24dd4ba25166dac84e0a.png"
new_data_path = tf.keras.utils.get_file('new_data', origin=new_data_url)

img = tf.keras.utils.load_img(
    new_data_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "Keras: This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

"""
12-2. TensorFlow Lite モデルを実行する
"""
# Interpreter を使用してモデルを読み込みます。
TF_MODEL_FILE_PATH = f'{models_dir}/model_{file_name}.tflite' # The default path to the saved TensorFlow Lite model

interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)

# 変換されたモデルからシグネチャを出力して、入力 (および出力) の名前を取得します。
print(interpreter.get_signature_list())

# 次のようにシグネチャ名を渡すことで、tf.lite.Interpreter.get_signature_runner を使用してサンプル画像で推論を実行し、読み込まれた TensorFlow モデルをテストできます。
classify_lite = interpreter.get_signature_runner('serving_default')
classify_lite

# 読み込まれた TensorFlow Lite モデル （predictions_lite）の最初の引数 （'inputs' の名前）に渡し、ソフトマックス活性化を計算し、計算された確率が最も高いクラスの予測を出力します。
predictions_lite = classify_lite(keras_tensor=img_array)['output_0']
score_lite = tf.nn.softmax(predictions_lite)

print(
    "TensorFlow Lite: This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
)

# Lite モデルが生成した予測は、元のモデルが生成した予測とほぼ同一になります。
print(np.max(np.abs(predictions - predictions_lite)))
