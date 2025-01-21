import React, { useState } from 'react';
import * as tf from '@tensorflow/tfjs';

const ImageClassifier: React.FC = () => {
  const host = import.meta.env.DEV ? 'http://192.168.11.2:5173/' : 'https://maitaketeikoku.github.io/poke_scan';

  // 利用可能なモデルのリスト
  const modelOptions = [
    {
      name: 'teachable Machine',
      url: `${host}/models/poke_scan_test/model.json`,
    },
    {
      name: 'model_3_15epochs',
      url: `${host}/models/model_3_15epochs_tfjs/model.json`,
    },
  ];

  const [predictions, setPredictions] = useState<string | null>(null);
  const [image, setImage] = useState<string | null>(null); // アップロードした画像
  const [model, setModel] = useState<tf.LayersModel | null>(null); // モデル
  const [selectedModel, setSelectedModel] = useState<string>(modelOptions[0].url); // 選択されたモデル

  // モデルの読み込み
  const loadModel = async (modelUrl: string) => {
    try {
      const loadedModel = await tf.loadLayersModel(modelUrl);
      setModel(loadedModel);
      console.log(`モデル "${modelUrl}" が正常に読み込まれました`);
    } catch (error) {
      console.error('モデルの読み込み中にエラーが発生しました:', error);
    }
  };

  // モデル選択変更時の処理
  const handleModelChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const selectedUrl = e.target.value;
    setSelectedModel(selectedUrl);
    setModel(null); // 前のモデルをリセット
    if (selectedUrl) {
      loadModel(selectedUrl);
    }
  };

  // 画像アップロード処理
  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImage(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  // 画像分類処理
  const classifyImage = async () => {
    if (model && image) {
      const imgElement = document.createElement('img');
      imgElement.src = image;
      imgElement.onload = async () => {
        try {
          const tensor = tf.browser
            .fromPixels(imgElement)
            .resizeNearestNeighbor([224, 224]) // モデルの入力サイズにリサイズ
            .expandDims(0)
            .toFloat()
            .div(tf.scalar(255)); // ピクセル値を正規化

          const prediction = (await model.predict(tensor)) as tf.Tensor;
          const predictedClass = prediction.argMax(-1).dataSync()[0]; // クラスを取得
          setPredictions(`Predicted Pokémon: ${predictedClass}`);
        } catch (error) {
          console.error('分類中にエラーが発生しました:', error);
        }
      };
    }
  };

  return (
    <div>
      <h1>ポケモンクラス分類器</h1>

      {/* モデル選択 */}
      <select id="model-select" value={selectedModel} onChange={handleModelChange}>
        <option value="">モデルを選択してください</option>
        {modelOptions.map((option) => (
          <option key={option.url} value={option.url}>
            {option.name}
          </option>
        ))}
      </select>

      {/* 画像アップロード */}
      <input type="file" accept="image/*" onChange={handleImageChange} />
      {image && <img src={image} alt="Uploaded" width="150" />}

      {/* 分類ボタン */}
      <button onClick={classifyImage} disabled={!model || !image}>
        分類を実行
      </button>

      {/* 結果表示 */}
      <div>{predictions ? predictions : 'モデルを選択してください...'}</div>
    </div>
  );
};

export default ImageClassifier;
