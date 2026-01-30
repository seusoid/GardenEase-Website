# Plant detection model

This folder holds the **YOLO model weights** used for plant identification in GardenEase.

- The app expects a file named **`best.pt`** (trained with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)).
- `best.pt` is not included in the repo because of its size. To run plant detection locally:
  1. Train or obtain a YOLO model and export `best.pt`.
  2. Place `best.pt` in this folder (`backend/plant_model/`).

The plant detection **code** (API, UI, and logic) lives in **`backend/plant_detection/`**.
