import React, { useState, useEffect } from "react";
import Webcam from "react-webcam";
import { ChartDataSets } from "chart.js";
import { Line } from "react-chartjs-2";
import {
  image,
  browser,
  tidy,
  dispose,
  cumsum,
  reshape,
} from "@tensorflow/tfjs";
import tensorStore from "../lib/tensorStore";
import Preprocessor from "../lib/preprocessor";
import Posprocessor from "../lib/posprocessor";

const postprocessor = new Posprocessor(tensorStore);
const Fili = require("fili");
const preprocessor = new Preprocessor(tensorStore, postprocessor);
const config: ChartDataSets = {
  fill: false,
  lineTension: 0.1,
  borderDash: [],
  borderDashOffset: 0.0,
  pointRadius: 0,
};

// When use props
type CamProps = {
  name?: string;
};
type GraphProps = {
  labels: string[];
  rppg: number[];
};

function WebcamPage({ name }: CamProps) {
  const webcamRef = React.useRef<any>(null);
  const intervalId = React.useRef<NodeJS.Timeout>();
  const coutdownIntervalId = React.useRef<NodeJS.Timeout>();

  const [isRecording, setRecording] = useState(false);
  const [countDown, setCountDown] = useState(30);
  const [charData, setCharData] = useState<GraphProps>({
    labels: [],
    rppg: [],
  });

  const refCountDown = React.useRef(30);

  const capture = React.useCallback(() => {
    if (webcamRef.current !== null) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc === null) return;
      const img = new Image(480, 640);

      img.src = imageSrc;
      img.onload = () => {
        const origVExpand: any = tidy(() =>
          browser.fromPixels(img).expandDims(0)
        );
        const crop = image.cropAndResize(
          origVExpand,
          [[0.1, 0.3, 0.56, 0.7]],
          [0],
          [36, 36],
          "bilinear"
        );
        dispose(origVExpand);
        const origV: any = crop.reshape([36, 36, 3]);
        tensorStore.addRawTensor(origV);
      };
    }
  }, [webcamRef]);

  const plotData = {
    labels: charData.labels,
    datasets: [
      {
        ...config,
        label: "Pulse",
        borderColor: "red",
        data: charData.rppg,
      },
    ],
  };

  const plotGraph = () => {
    const pltData = tensorStore.rppgPltData;
    const iirCalculator = new Fili.CalcCascades();
    const iirFilterCoeffs = iirCalculator.bandpass({
      order: 1, // cascade 3 biquad filters (max: 12)
      characteristic: "butterworth",
      Fs: 30, // sampling frequency
      Fc: 1.375, // (2.5-0.75) / 2 + 0.75, 2.5 --> 150/60, 0.75 --> 45/60 # 1.625
      BW: 1.25, // 2.5 - 0.75 = 1.75
      gain: 0, // gain for peak, lowshelf and highshelf
      preGain: false, // adds one constant multiplication for highpass and lowpass
    });
    const iirFilter = new Fili.IirFilter(iirFilterCoeffs);
    if (pltData) {
      const rppgCumsum = cumsum(reshape(pltData, [-1, 1]), 0).dataSync();
      const result = iirFilter
        .filtfilt(rppgCumsum)
        .slice(0, rppgCumsum.length - 60);
      const labels = Array.from(pltData.keys())
        .map((i) => i.toString())
        .slice(0, rppgCumsum.length - 60);
      setCharData({
        labels,
        rppg: result,
      });
    }
  };

  const startRecording = async () => {
    await postprocessor.loadModel();
    intervalId.current = setInterval(capture, 30);
    coutdownIntervalId.current = setInterval(() => {
      setCountDown((prevCount) => prevCount - 1);
      refCountDown.current -= 1;
      if (refCountDown.current === 0) {
        plotGraph();
        stopRecording();
      }
    }, 1000);
    setRecording(true);
    preprocessor.startProcess();
  };

  const stopRecording = () => {
    if (intervalId.current) {
      clearInterval(intervalId.current);
    }
    if (coutdownIntervalId.current) {
      clearInterval(coutdownIntervalId.current);
    }
    preprocessor.stopProcess();
    tensorStore.reset();
    setCountDown(30);
    refCountDown.current = 30;
    setRecording(false);
  };

  useEffect(
    () => () => {
      if (intervalId.current) {
        clearInterval(intervalId.current);
      }

      if (coutdownIntervalId.current) {
        clearInterval(coutdownIntervalId.current);
      }
    },
    []
  );

  useEffect(
    () => () => {
      preprocessor.stopProcess();
      // postprocessor.stopProcess();
      tensorStore.reset();
    },
    []
  );

  return (
    <div
      className="WebcamPageContainer"
      style={{
        position: "fixed",
        display: "flex",
        left: "20%",
        top: "8%",
      }}
    >
      <div>
        <h1>WebCam Demo</h1>
        {!isRecording && (
          <button
            className="RecordButton"
            onClick={startRecording}
            type="button"
          >
            Start the Demo
          </button>
        )}
        <Webcam
          width={500}
          height={500}
          mirrored
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
        />
        <p
          className="countDown"
          style={{
            fontSize: "30px",
          }}
        >
          {countDown}
        </p>
        {!isRecording && !!charData.rppg.length && (
          <Line
            data={plotData}
            width={1200}
            height={300}
            options={{
              responsive: false,
              animation: {
                duration: 0,
              },
              scales: {
                yAxes: [
                  {
                    ticks: {
                      display: false,
                    },
                  },
                ],
                xAxes: [
                  {
                    ticks: {
                      display: false,
                    },
                  },
                ],
              },
            }}
          />
        )}
      </div>
    </div>
  );
}

export default WebcamPage;
