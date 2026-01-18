# Face Detection Restore - ComfyUI 节点

这是一个为 ComfyUI 设计的人脸检测和恢复工具包，支持批量处理、多种检测器（MediaPipe & YOLO+ViTPose）以及高质量的人脸贴回功能。

## 资源下载

* **示例工作流**：[夸克网盘下载](https://pan.quark.cn/s/775e96674960)
* **模型文件**：[夸克网盘下载](https://pan.quark.cn/s/bae26a0d1f51)
* 
## 模型放置路径

请将下载的文件按以下结构放置：

* `ComfyUI/models/mediapipe/`
  * `face_landmarker.task`
  * `selfie_multiclass_256x256.tflite`
* `ComfyUI/models/detection/`
  * `vitpose-l-wholebody.onnx`
  * `yolov10m.onnx`

## 安装方法

1. 将本项目文件夹放入 ComfyUI 的 `custom_nodes` 目录中。
2. 在终端进入该目录，运行：

   ```bash
   pip install -r requirements.txt
   ```

## 包含节点

1. **BatchFaceDetectionModelLoader**：加载 YOLO 和 ViTPose 检测模型。
2. **BatchFaceDetection**：批量检测人脸，输出裁剪后的图像、掩码及边界框。
3. **BatchFaceRestoreToSource**：将编辑后的人脸贴回原图，支持 `MASK`、`全局`等混合模式贴回。

