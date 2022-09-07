# Stereo Depth Estimation

Depth estimation by applying triangulation on image pairs from a stereo camera.

Use main branch as basis for all further code.

## Steps to use
- Download the following files from the Perception folder into the path given below:
```
data.yaml -> /object_detect/yolov5/models/data.yaml
last.pt -> /object_detect/yolov5/weights/last.pt
```

- Following weights already exist in the repository.
```
23_loss_0.38.pt -> /Keypoint_Detection/Weights/23_loss_0.38.pt
best_keypoints_8080.onnx -> /Keypoint_Detection/Weights/best_keypoints_8080.onnx
```

## Execution
### Currently(5 June 2022) working:

- Create venv in VSCode 
- Run `pip install -r requirements.txt`
- Make sure you have cloned/unzipped yolov5 and yolo-tensorrt repositories in object_detect
- Make sure to edit the main.py to specify the path of the image/video you're going to run the code on
- Run `main.py` - implemented uptil keypoint detection and visualization - faulty propagation.
- In `view_utils.py` and `disparity.py`, find development progress of template registration for stereo matching, refere issues.
