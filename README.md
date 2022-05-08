## yolov5_ros
Colconized [yolov5](https://github.com/ultralytics/yolov5) Python library for easier inference deployment on a ROS2 system.

## Installation

    pip3 install torch torchvision
    cd <your_colcon_ws>
    rosdep install --from-paths src -i -r -y
    colcon build

## Usage

Import inference from yolov5_ros and call the `predict` function.

    import matplotlib

    model = Yolov5(
        'test.pt',
        'test.yaml',
        conf_thres=0.4,
        iou_thres=0.5,
        device=0
    )

    image = image.imread('test.jpg')
    classes, bounding_boxes = model.predict(image)
