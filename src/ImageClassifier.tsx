import React, { useRef, useState, useMemo, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import {
  Container, HStack, Box,
  IconButton, Select, Option,
  useNotice,
} from "@yamada-ui/react";
import { PlayIcon, PauseIcon } from "@yamada-ui/lucide";
import { BarChart, BarProps } from "@yamada-ui/charts";

// 閾値
const threshold = 0.9;
// 表示する上位のクラス数
const topCount = 5;

const url = `${import.meta.env.BASE_URL}models`;
const pokedex_bg = `${import.meta.env.BASE_URL}pokedex_bg.png`;

// 利用可能なモデルのリスト
const models = [
  {
    name: '151classes_8-15epochs',
    url: `/151classes_8-15epochs_tfjs`,
    shape: 224
  },
  {
    name: '1025classes_10epochs',
    url: `/1025classes_10epochs_tfjs`,
    shape: 180
  },
];

const ImageClassifier: React.FC = () => {
  const notice = useNotice({
    limit: 1,
    duration: null,
    isClosable: true,
    placement: "bottom",
  });

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const intervalRef = useRef<number | null>(null);

  const [isError, setIsError] = useState<boolean>(false);

  const [isLoadingCamera, setIsLoadingCamera] = useState<boolean>(true);
  const [videoDevices, setVideoDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);

  const [isLoadingModel, setIsLoadingModel] = useState<boolean>(true);
  const [selectedModelUrl, setSelectedModelUrl] = useState<string>(models[0].url);
  const [model, setModel] = useState<tf.LayersModel | null>(null);

  const [predictions, setPredictions] = useState<{ name: number; value: number }[]>([]);

  const series: BarProps[] = useMemo(
    () =>
      [
        { dataKey: "value", color: "primary.500" },
      ],
    [],
  );

  // カメラの対応確認
  useEffect(() => {
    (async () => {
      setIsLoadingCamera(true);

      if ('mediaDevices' in navigator && 'getUserMedia' in navigator.mediaDevices) {
        // カメラのデバイス一覧を取得
        const devices = await navigator.mediaDevices.enumerateDevices();
        // ビデオデバイスのみを抽出し、背面カメラを優先してソート
        const newVideoDevices = devices
          .filter((device) => device.kind === "videoinput")
          .sort((a, b) => {
            const aIsBack = a.label.includes("背面") || a.label.toLowerCase().includes("back");
            const bIsBack = b.label.includes("背面") || b.label.toLowerCase().includes("back");

            if (aIsBack && !bIsBack) return -1;
            if (!aIsBack && bIsBack) return 1;
            return 0;
          });
        setVideoDevices(newVideoDevices);
        setSelectedDeviceId(newVideoDevices[0]?.deviceId);

        setIsLoadingCamera(false);
      } else {
        setIsError(true);
        notice({
          title: "カメラ非対応",
          description: "このブラウザはカメラに対応していないよ",
          status: "warning",
        });
      }
    })();
  }, []);

  // モデルの読み込み
  useEffect(() => {
    (async () => {
      setIsLoadingModel(true);
      notice({
        title: "AIモデル読込中",
        description: "AIモデルを読み込んでいるよ...",
        status: "info",
      });

      try {
        const modelUrl = `${url}${selectedModelUrl}/model.json`;
        const loadedModel = await tf.loadLayersModel(modelUrl);
        setModel(loadedModel);

        notice({
          title: "AIモデル読込完了",
          description: "AIモデルの読み込みが完了したよ！カメラを選んでスタートしてね！",
          status: "success",
        });
        setIsLoadingModel(false);
      } catch (e) {
        setIsError(true);
        notice({
          title: "AIモデル読込失敗",
          description: `AIモデルの読み込みに失敗しました：${e}`,
          status: "success",
        });
      }
    })();
  }, [selectedModelUrl]);

  // カメラを起動し、分類を開始
  const start = async () => {
    setIsCameraActive(true);

    if (videoRef.current) {
      try {
        const selectedDevice = videoDevices.find((device) => device.deviceId === selectedDeviceId);
        if (!selectedDevice) {
          notice({
            title: "カメラ選択失敗",
            description: "選択されたカメラが見つかりませんでした。",
            status: "warning",
          });
          return;
        }

        const stream = await navigator.mediaDevices.getUserMedia({
          video: { deviceId: { exact: selectedDevice.deviceId } },
        });
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      } catch (e) {
        notice({
          title: "カメラ起動失敗",
          description: `Webカメラの起動に失敗しました：${e}`,
          status: "error",
        });
        return;
      }
    }

    if (intervalRef.current === null) {
      intervalRef.current = window.setInterval(predict, 100);
    }
  };

  // 動画の分類
  const predict = async () => {
    if (model && videoRef.current) {
      const video = videoRef.current;

      const shape: number = models.find((model) => model.url === selectedModelUrl)?.shape || 224;

      try {
        const tensor = tf.browser
          .fromPixels(video)
          .resizeNearestNeighbor([shape, shape])
          .expandDims(0)
          .toFloat()
          .div(tf.scalar(255));

        const prediction = (await model.predict(tensor)) as tf.Tensor;
        const predictionData = prediction.dataSync();
        const predictionProbabilities = tf.softmax(tf.tensor(predictionData)).dataSync();
        const topK = Array.from(predictionProbabilities)
          .map((prob, idx) => ({ id: idx, prob }))
          .sort((a, b) => b.prob - a.prob)
          .slice(0, topCount);
        const results = topK.map(
          (item) => {
            return {
              name: item.id,
              value: item.prob * 100
            };
          });
        setPredictions(results);

        tensor.dispose();
        prediction.dispose();
      } catch (e) {
        notice({
          title: "AIモデル分類失敗",
          description: `AIモデルの分類中にエラーが発生しました：${e}`,
          status: "error",
        });
      }
    }
  };

  // 分類を停止
  const stop = () => {
    if (intervalRef.current !== null) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    // カメラのストリームを停止
    if (videoRef.current && videoRef.current.srcObject instanceof MediaStream) {
      const stream = videoRef.current.srcObject;
      const tracks = stream.getTracks();
      tracks.forEach((track) => track.stop());
    }

    // カメラのストリームを解除
    videoRef.current && (videoRef.current.srcObject = null);

    setIsCameraActive(false);
  };

  return (
    <Container centerContent h="100dvh">
      <HStack w="full" maxW="xl">
        <Select
          value={selectedDeviceId}
          onChange={setSelectedDeviceId}
          isDisabled={isCameraActive}
          defaultValue={videoDevices[0]?.deviceId}
          placeholderInOptions={false}
          whiteSpace='nowrap'
          overflow='hidden'
          textOverflow='ellipsis'
        >
          {videoDevices.map((device) => (
            <Option key={device.deviceId} value={device.deviceId}>
              {device.label || "不明なカメラ"}
            </Option>
          ))}
        </Select>

        <Select
          value={selectedModelUrl}
          onChange={setSelectedModelUrl}
          isDisabled={isCameraActive}
          defaultValue={models[0].url}
          placeholderInOptions={false}
          whiteSpace='nowrap'
          overflow='hidden'
          textOverflow='ellipsis'
        >
          {models.map((model) => (
            <Option key={model.url} value={model.url}>
              {model.name || "不明なモデル"}
            </Option>
          ))}
        </Select>

        <IconButton
          icon={isCameraActive ? <PauseIcon /> : <PlayIcon />}
          onClick={isCameraActive ? stop : start}
          isDisabled={isError || isLoadingCamera || isLoadingModel}
          colorScheme="primary"
        />
      </HStack>

      <Box
        rounded="md"
        backgroundImage={pokedex_bg}
        backgroundSize="cover"
        backgroundRepeat="no-repeat"
        bgColor="red.500"
        width="100%"
        maxW="xl"
        p="2"
      >
        <video ref={videoRef}
          width="100%" height="auto" autoPlay muted
          style={{
            marginTop: "45%",
            backgroundColor: "#1c1c1c",
            border: "4px solid #d3d5da",
            borderRadius: "12px"
          }}
        />
      </Box>

      <Box width="100%" maxW="xl">
        <BarChart
          data={predictions}
          series={series}
          dataKey="name"
          size="sm"
          unit="%"
          yAxisProps={{ domain: [0, 100], tickCount: 6 }}
          gridAxis="x"
          withTooltip={false}
          referenceLineProps={[{ y: threshold * 100, color: "red.500" }]}
        />
      </Box>

      <div>
        {predictions.map((result, index) => (
          <div key={index}>{result.name}: {result.value}</div>
        ))}
      </div>
    </Container>
  );
};

export default ImageClassifier;
