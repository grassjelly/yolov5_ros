import os
from glob import glob
from setuptools import setup

package_name = 'yolov5_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=[
        package_name,
        'yolov5_ros.yolov5',
        'yolov5_ros.yolov5.models',
        'yolov5_ros.yolov5.utils'
    ],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools', 'shapely'],
    zip_safe=True,
    maintainer='Juan Miguel jimeno',
    maintainer_email='jimenojmm@gmail.com',
    description='Colconized yolov5_ros',
    license='GPL3.0',
)
