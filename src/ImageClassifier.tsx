import React, { useState } from 'react';
import * as tf from '@tensorflow/tfjs';

const ImageClassifier: React.FC = () => {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<string | null>(null);

  // クラス名を設定 (トレーニング時のクラス名を順番通りに指定)
  const classNames = ['1', '4', '7']; // 実際のクラス名に置き換えてください

  const loadModelAndPredict = async (imageElement: HTMLImageElement) => {
    try {
      // モデルを読み込む
      const model = await tf.loadGraphModel('/models/model_3_15epochs_tfjs/model.json');

      // 画像をTensorに変換
      const tensor = tf.browser
        .fromPixels(imageElement)
        .resizeNearestNeighbor([180, 180]) // モデルの入力サイズに合わせる
        .toFloat()
        .div(255.0) // 正規化
        .expandDims();

      // 推論
      const predictions = (await model.predict(tensor)) as tf.Tensor;
      const scores = predictions.softmax().dataSync();

      // 最大スコアのクラスを取得
      const maxIndex = scores.indexOf(Math.max(...scores));
      setPrediction(`${classNames[maxIndex]} (${(scores[maxIndex] * 100).toFixed(2)}%)`);
    } catch (error) {
      console.error('エラーが発生しました:', error);
    }
  };

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        setImageSrc(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div>
      <h1>Image Classifier</h1>
      <input type="file" accept="image/*" onChange={handleImageUpload} />
      {imageSrc && (
        <div>
          <img
            src={imageSrc}
            alt="Uploaded"
            onLoad={(e) => loadModelAndPredict(e.currentTarget)}
            style={{ maxWidth: '300px', maxHeight: '300px' }}
          />
        </div>
      )}
      {prediction && <p>Prediction: {prediction}</p>}
    </div>
  );
};

export default ImageClassifier;
