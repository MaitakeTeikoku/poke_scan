# 画像の入ったimages.zipファイルと、google_colab.pyファイルを、Google Driveにアップロードしておく。

!pip install tensorflowjs
!mkdir /content/poke_scan
!rm -rf /content/poke_scan/images
!unzip /content/drive/MyDrive/development/poke_scan/images.zip -d /content/poke_scan
!cp /content/drive/MyDrive/development/poke_scan/google_colab.py /content/poke_scan
!python /content/poke_scan/google_colab.py
