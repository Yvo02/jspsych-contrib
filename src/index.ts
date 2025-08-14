import "@mediapipe/face_mesh";
import {
  FaceLandmarker,
  FilesetResolver,
  FaceLandmarkerResult,
} from "@mediapipe/tasks-vision";

import autoBind from "auto-bind";
import { JsPsych, JsPsychExtension, JsPsychExtensionInfo } from "jspsych";
import { Euler, Matrix4, Vector3 } from "three";

interface IFaceTrackingResult {
  frame_id: number;
  transformation?: number[];
  rotation?: Euler;
  translation?: Vector3;
  blendshapes?: Array<{ name: string; score: number }>;
}

class MediapipeFacemeshExtension implements JsPsychExtension {
  static info: JsPsychExtensionInfo = {
    name: "mediapipe-face-mesh",
  };

  private recordedChunks = new Array<any>();
  private animationFrameId: number;
  public mediaStream: MediaStream;
  private videoElement: HTMLVideoElement;
  private canvasElement: HTMLCanvasElement;

  // Alte API
  private faceMesh: FaceMesh;
  // Neue API
  private faceLandmarker: FaceLandmarker;
  private usingNewAPI = false;

  private onResultCallbacks = new Array<(ITrackingResult) => void>();
  private recordTracks = false;

  constructor(private jsPsych: JsPsych) {
    autoBind(this);
  }

  initialize = async (params): Promise<void> => {
    this.usingNewAPI = params?.useFaceLandmarker ?? false;

    if (this.usingNewAPI) {
      // Neue Face Landmarker API
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );
      this.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        },
        runningMode: "VIDEO", // wichtig für Video-Modus!
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
        numFaces: 1,
      });
    } else {
      // Alte FaceMesh API
      this.faceMesh = new FaceMesh({
        locateFile:
          params?.locateFile ??
          function (file) {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
          },
      });

      this.faceMesh.setOptions({
        maxNumFaces: 1,
        enableFaceGeometry: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
        refineLandmarks: false,
      });

      await this.faceMesh.initialize();
    }
  };

  on_start = (): void => {
    this.canvasElement?.remove();
    this.videoElement?.remove();

    this.canvasElement = document.createElement("canvas");
    this.canvasElement.width = 1280;
    this.canvasElement.height = 720;

    this.videoElement = document.createElement("video");
    this.videoElement.muted = true;

    this.mediaStream = this.jsPsych.pluginAPI.getCameraStream();

    if (!this.mediaStream) {
      console.warn("Camera not initialized.");
      return;
    }

    this.videoElement.srcObject = this.mediaStream;

    this.videoElement.onloadedmetadata = () => {
      this.stopAnimationFrame();
      this.animationFrameId = window.requestAnimationFrame(
        this.processFrame.bind(this)
      );
    };

    if (!this.usingNewAPI) {
      this.faceMesh.onResults(this.onMediaPipeResult.bind(this));
    }

    this.videoElement.play();
  };

  on_load = (params) => {
    this.recordedChunks = [];
    this.recordTracks = params?.record ?? false;
  };

  on_finish = () => {
    console.log("Tracked chunks: " + this.recordedChunks.length);
    this.stopAnimationFrame();
    this.recordTracks = false;
    return { face_mesh: this.recordedChunks };
  };

  private stopAnimationFrame(): void {
    window.cancelAnimationFrame(this.animationFrameId);
  }

  private async processFrame(): Promise<void> {
    const ctx = this.canvasElement.getContext("2d");
    ctx.drawImage(this.videoElement, 0, 0);

    if (this.usingNewAPI) {
      const results: FaceLandmarkerResult =
        this.faceLandmarker.detectForVideo(
          this.videoElement,
          performance.now()
        );
      this.onFaceLandmarkerResult(results);
    } else {
      await this.faceMesh.send({ image: this.canvasElement });
    }

    this.animationFrameId = window.requestAnimationFrame(
      this.processFrame.bind(this)
    );
  }

  public addTrackingResultCallback(callback: (ITrackingResult) => void) {
    this.onResultCallbacks.push(callback);
  }

  public removeTrackingResultCallback(callback: (ITrackingResult) => void) {
    this.onResultCallbacks.splice(this.onResultCallbacks.indexOf(callback), 1);
  }

  // Alte API
  private onMediaPipeResult(results: Results): void {
    if (results.multiFaceGeometry[0]) {
      const transformationMatrix = results.multiFaceGeometry[0]
        .getPoseTransformMatrix()
        .getPackedDataList();
      const rotation = new Euler().setFromRotationMatrix(
        new Matrix4().fromArray(transformationMatrix)
      );
      const translation = new Vector3().setFromMatrixPosition(
        new Matrix4().fromArray(transformationMatrix)
      );

      const result: IFaceTrackingResult = {
        frame_id: this.animationFrameId,
        transformation: transformationMatrix,
        rotation,
        translation,
      };

      if (this.recordTracks) this.recordedChunks.push(result);
      this.onResultCallbacks.forEach((cb) => cb(result));
    }
  }

  // Neue API → Unity-kompatibles JSON
  private onFaceLandmarkerResult(results: FaceLandmarkerResult): void {
    let rotation = new Euler();
    let translation = new Vector3();

    if (results.facialTransformationMatrixes?.length) {
      const transformationMatrix = results.facialTransformationMatrixes[0].data;
      rotation.setFromRotationMatrix(new Matrix4().fromArray(transformationMatrix));
      translation.setFromMatrixPosition(new Matrix4().fromArray(transformationMatrix));
    }

    let landmarksFlat: number[] = [];
    if (results.landmarks?.length) {
      landmarksFlat = results.landmarks[0].flatMap(lm => [lm.x, lm.y, lm.z]);
    }

    const blendshapesObj: Record<string, { score: number }> = {};
    if (results.faceBlendshapes?.length) {
      results.faceBlendshapes[0].categories.forEach(cat => {
        blendshapesObj[cat.categoryName] = { score: cat.score };
      });
    }

    const orientation = "front";

    const unityData = {
      landmarks: landmarksFlat,
      orientation,
      blendshapes: blendshapesObj,
      rotation: {
        x: -rotation.x, // X invertieren
        y: rotation.y,
        z: rotation.z
      },
      translation: {
        x: translation.x / 100,
        y: translation.y / 100,
        z: translation.z / 100
      }
    };

    if (this.recordTracks) this.recordedChunks.push(unityData);
    this.onResultCallbacks.forEach(cb => cb(unityData));
  }
}

export default MediapipeFacemeshExtension;
