# Local pose model assets

`pose_landmarker_lite.task` is the MediaPipe Pose Landmarker Lite float16 model, version 1, obtained from Google's published model storage:
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

The WASM/JS runtime is copied from the pinned `@mediapipe/tasks-vision` npm package during `prebuild`/`predev`; it is not downloaded at runtime. The model is a detection dependency, never a replacement recording or trajectory. No user video is shipped here.
