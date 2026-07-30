from setuptools import find_packages, setup

package_name = 'circumnavigation_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/sim_launch.py',
            'launch/webcam_launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mitchell',
    maintainer_email='mitch.torok@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    
    entry_points={
        'console_scripts': [
            'bearing_measurement_generation = circumnavigation_controller.bearing_measurement_generation:main',
            'controller = circumnavigation_controller.controller:main',
            'relative_position_controller = circumnavigation_controller.relative_position_controller:main',
            'camera_calibrate = circumnavigation_controller.camera_calibrate:main',
            'aruco_detector = circumnavigation_controller.aruco_detector:main',
            'cinematic_gui = circumnavigation_controller.cinematic_gui:main'
        ],
    },
)
