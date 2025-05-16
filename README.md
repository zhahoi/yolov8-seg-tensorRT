# yolov8-seg-tensorRT
在Visual Studio上部署TensorRT版本的yolov8-seg（**TensorRT10.x**)。



### 配置环境

- OpenCV 3.4.10
- TensorRT-10.6
- Visual Studio 2019

**注：可以参考自己电脑CUDA版本选择TensorRT。TensorRT部署需要保证模型转换和推理保持一致。**



### 推理设置

（1）现在Visual Studio新建一个空白工程，将本仓库代码放到该工程中。

![微信截图_20250516084350](C:\Users\HIT-HAYES\Desktop\微信截图_20250516084350.png)

（2）在工程中载入推理需要依赖的库。

![微信截图_20250516084012](C:\Users\HIT-HAYES\Desktop\微信截图_20250516084012.png)

![微信截图_20250516084108](C:\Users\HIT-HAYES\Desktop\微信截图_20250516084211.png)

![微信截图_20250516084306](C:\Users\HIT-HAYES\Desktop\微信截图_20250516084306.png)

（3）选择开始执行，应该在工程的**Release**的目录下可以成功地生成`.exe`文件。

![微信截图_20250516085457](C:\Users\HIT-HAYES\Desktop\微信截图_20250516085457.png)

（4）执行推理。

![微信截图_20250516085552](C:\Users\HIT-HAYES\Desktop\微信截图_20250516085552.png)

推理结果：

![微信截图_20250516085620](C:\Users\HIT-HAYES\Desktop\微信截图_20250516085620.png)



### 写在后面

本仓库代码仅仅是为了学习TensorRT部署而创建，推理代码来自[YOLOv8-TensorRT](https://github.com/triple-Mu/YOLOv8-TensorRT)。由于本电脑的TensorRT版本和参考推理代码之间存在差异，自己各种搜索之后使其成功运行。上传到Github可以供大家参考，少走弯路。以上。



### Reference

-[YOLOv8-TensorRT](https://github.com/triple-Mu/YOLOv8-TensorRT)
