## yolov5_ros
Colconized [yolov5](https://github.com/ultralytics/yolov5) Python library for easier inference deployment on a ROS2 system.

## Installation

    pip3 install torch torchvision
    cd <your_colcon_ws>
    rosdep install --from-paths src -i -r -y
    colcon build

