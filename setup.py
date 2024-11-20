from setuptools import setup

package_name = 'rosbag_utils'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jerome',
    maintainer_email='jerome@idsia.ch',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'read = rosbag_utils.reader:main',
            'video = rosbag_utils.h264_video:main',
            'hdf5 = rosbag_utils.export_h5df:main',
            'hdf5_sync = rosbag_utils.export_h5df_sync:main',
            'csv_kinect = rosbag_utils.export_csv_kinect:main',
            'csv_offering = rosbag_utils.export_csv_offering:main',
            'dataset_descriptor_creator = rosbag_utils.dataset_description_creator:main',
            'hdf5_lidar_human_pose_estimation = rosbag_utils.export_hdf5_lidar_human_pose_estimation:main',
        ],
    },
)
