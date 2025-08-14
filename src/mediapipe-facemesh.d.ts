export {};

declare global {
  export interface NormalizedLandmark {
    x: number;
    y: number;
    z: number;
    visibility?: number;
  }
  export type NormalizedLandmarkList = NormalizedLandmark[];
  export type NormalizedLandmarkListList = NormalizedLandmarkList[];

  export interface MatrixData {
    getPackedDataList(): number[];
  }
  export interface FaceGeometry {
    getPoseTransformMatrix(): MatrixData;
  }
  export interface Results {
    multiFaceLandmarks: NormalizedLandmarkListList;
    multiFaceGeometry: FaceGeometry[];
  }
  class FaceMesh {
    constructor(config?: any);
    initialize(): Promise<void>;
    onResults(listener: (results: Results) => void): void;
    send(inputs: { image: HTMLCanvasElement }): Promise<void>;
    setOptions(options: any): void;
  }
}

declare module "@mediapipe/tasks-vision" {
  export interface BlendshapeCategory {
    categoryName: string;
    score: number;
  }
  export interface Blendshape {
    categories: BlendshapeCategory[];
  }
  export interface FacialTransformationMatrix {
    data: number[];
  }
  export interface FaceLandmarkerResult {
    faceBlendshapes?: Blendshape[];
    facialTransformationMatrixes?: FacialTransformationMatrix[];
    landmarks?: NormalizedLandmarkList[];
  }
  export class FaceLandmarker {
    static createFromOptions(
      vision: any,
      options: any
    ): Promise<FaceLandmarker>;
    detectForVideo(video: HTMLVideoElement, timestamp: number): FaceLandmarkerResult;
  }
  export class FilesetResolver {
    static forVisionTasks(path: string): Promise<any>;
  }
}
