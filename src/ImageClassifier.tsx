import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';

const ImageClassifier: React.FC = () => {
  const host = import.meta.env.DEV ? 'http://192.168.11.2:5173' : 'https://maitaketeikoku.github.io/poke_scan';

  // 利用可能なモデルのリスト
  const modelOptions = [
    {
      name: 'poke_scan_types',
      url: `${host}/models/poke_scan_types/model.json`,
    },
    {
      name: 'poke_scan_test',
      url: `${host}/models/poke_scan_test/model.json`,
    },
    {
      name: 'model_3_15epochs',
      url: `${host}/models/model_3_15epochs_tfjs/model.json`,
    },
  ];

  // タイプ名の情報
  const typeNameList: { id: number, name: string, nameJp: string }[] = [
    { id: 1, name: "normal", nameJp: "ノーマル" },
    { id: 2, name: "fire", nameJp: "ほのお" },
    { id: 3, name: "water", nameJp: "みず" },
    { id: 4, name: "grass", nameJp: "くさ" },
    { id: 5, name: "electric", nameJp: "でんき" },
    { id: 6, name: "ice", nameJp: "こおり" },
    { id: 7, name: "fighting", nameJp: "かくとう" },
    { id: 8, name: "poison", nameJp: "どく" },
    { id: 9, name: "ground", nameJp: "じめん" },
    { id: 10, name: "flying", nameJp: "ひこう" },
    { id: 11, name: "psychic", nameJp: "エスパー" },
    { id: 12, name: "bug", nameJp: "むし" },
    { id: 13, name: "rock", nameJp: "いわ" },
    { id: 14, name: "ghost", nameJp: "ゴースト" },
    { id: 15, name: "dragon", nameJp: "ドラゴン" },
    { id: 16, name: "dark", nameJp: "あく" },
    { id: 17, name: "steel", nameJp: "はがね" },
    { id: 18, name: "fairy", nameJp: "フェアリー" }
  ];

  const [predictions, setPredictions] = useState<string[]>([]);
  const [image, setImage] = useState<string | null>(null); // アップロードした画像
  const [model, setModel] = useState<tf.LayersModel | null>(null); // モデル
  const [selectedModel, setSelectedModel] = useState<string>(modelOptions[0].url); // 選択されたモデル

  useEffect(() => {
    loadModel(selectedModel);
  }, []);

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
    setPredictions([]);
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

          if (selectedModel === modelOptions[0].url) {
            const predictionData = prediction.dataSync();

            // トップ3の予測結果を取得
            const topK = Array.from(predictionData)
              .map((prob, idx) => ({ id: idx, prob })) // インデックスと確率をペア化
              .sort((a, b) => b.prob - a.prob) // 確率で降順ソート
              .slice(0, 5); // トップ5を取得

            // 日本語名と確率を組み合わせて表示
            const results = topK.map(
              (item) =>
                `${typeNameList[item.id]?.nameJp || '未知'} : ${(item.prob * 100).toFixed(2)}%`
            );

            setPredictions(results); // 複数行で結果を表示
          } else {
            const predictedClass = prediction.argMax(-1).dataSync()[0]; // クラスを取得
            setPredictions([`Predicted Pokémon: ${predictedClass}`]);
          }
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
      <div>{predictions.map((result, index) => <div key={index}>{result}</div>)}</div>
    </div>
  );
};

export default ImageClassifier;
