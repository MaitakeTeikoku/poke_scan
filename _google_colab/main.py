"""
google_colab.pyを実行
"""
# 画像の入ったimages.zipファイルと、google_colab.pyファイルを、Google Driveにアップロードしておく。

!pip install tensorflowjs
!mkdir /content/poke_scan
!rm -rf /content/poke_scan/images
!unzip /content/drive/MyDrive/development/poke_scan/images.zip -d /content/poke_scan
!cp /content/drive/MyDrive/development/poke_scan/google_colab.py /content/poke_scan
!python /content/poke_scan/google_colab.py

!pip install tensorflow==2.15.0
!pip install tensorflow-decision-forests==1.8.1

import tensorflow as tf
tf.keras.backend.clear_session()

"""
sample.pyを実行
"""
# 画像の入ったimages.zipファイルと、sample.pyファイルを、Google Driveにアップロードしておく。

!pip install tensorflowjs
!mkdir /content/poke_scan
!rm -rf /content/poke_scan/images
!unzip /content/drive/MyDrive/development/poke_scan/images.zip -d /content/poke_scan
!cp /content/drive/MyDrive/development/poke_scan/sample.py /content/poke_scan
!python /content/poke_scan/sample.py

!pip install tensorflow==2.15.0
!pip install tensorflow-decision-forests==1.8.1

import tensorflow as tf
tf.keras.backend.clear_session()