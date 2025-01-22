import argparse
import h5py
import pathlib
import numpy as np
import os
import scipy.interpolate
from collections import OrderedDict
from functools import singledispatch
from typing import Any, List, Optional, Dict

import rclpy
import tf2_ros
import tf2_msgs.msg
import nav_msgs.msg
import sensor_msgs.msg
import geometry_msgs.msg
import visualization_msgs.msg
import azure_kinect_ros_msgs.msg

from rosbag_utils.reader import BagReader, header_stamp, sanitize

# Topics in bagfile:
"""
Topic information: Topic: /tf_static | Type: tf2_msgs/msg/TFMessage
Topic: /tf | Type: tf2_msgs/msg/TFMessage
Topic: /joint_states | Type: sensor_msgs/msg/JointState
Topic: /body_tracking_data | Type: azure_kinect_ros_msgs/msg/MarkerArrayStamped
Topic: /mobile_base_controller/odom | Type: nav_msgs/msg/Odometry
Topic: /scan_raw_back | Type: sensor_msgs/msg/LaserScan
Topic: /scan_raw | Type: sensor_msgs/msg/LaserScan 
Topic: /dlo_ros/odom | Type: nav_msgs/msg/Odometry
"""

# Extraction config
topics_names = [
    "/tf",
    "/tf_static",
    "/joint_states",
    "/body_tracking_data",
    "/mobile_base_controller/odom",
    "/scan_raw",
    "/scan_raw_back",
    "/dlo_ros/odom",
]

save_metadata_of_topics = ["/scan_raw", "/scan_raw_back"]

sync_dataset_on_topic = "/scan_raw_back"


frame_to_save_wrt = OrderedDict(
    {
        "base_laser_link": "base_link",
        "base_laser_back_link": "base_link",
        "azure_kinect_depth_camera_link": "base_link",
        "base_link": "odom",
    }
)


class ReaderException(Exception):
    pass


@singledispatch
def metadata_reader(msg: Any) -> np.ndarray:
    raise ReaderException(f'Cannot import metadata for message of type "{type(msg)}"')


@metadata_reader.register
def _(msg: sensor_msgs.msg.LaserScan) -> np.ndarray:
    return {
        "angle_min": msg.angle_min,
        "angle_max": msg.angle_max,
        "range_min": msg.range_min,
        "range_max": msg.range_max,
        "time_increment": msg.time_increment,
        "angle_increment": msg.angle_increment,
    }


@singledispatch
def reader(msg: Any) -> np.ndarray:
    raise ReaderException(f'Cannot import message of type "{type(msg)}"')


@reader.register
def _(msg: list) -> Optional[np.ndarray]:
    if msg is None or not msg:
        return None

    return np.stack([reader(e) for e in msg], axis=0)


@reader.register
def _(msg: geometry_msgs.msg.Vector3) -> np.ndarray:
    return np.array([msg.x, msg.y, msg.z], dtype=np.float32)


@reader.register
def _(msg: geometry_msgs.msg.Vector3Stamped) -> np.ndarray:
    return reader(msg.vector)


@reader.register
def _(msg: geometry_msgs.msg.Point32) -> np.ndarray:
    return reader.dispatch(geometry_msgs.msg.Point)(msg)


@reader.register
def _(msg: geometry_msgs.msg.Point) -> np.ndarray:
    return np.array([msg.x, msg.y, msg.z], dtype=np.float32)


@reader.register
def _(msg: geometry_msgs.msg.PointStamped) -> np.ndarray:
    return reader(msg.point)


@reader.register
def _(msg: geometry_msgs.msg.Quaternion) -> np.ndarray:
    return np.array([msg.x, msg.y, msg.z, msg.w], dtype=np.float32)


@reader.register
def _(msg: geometry_msgs.msg.QuaternionStamped) -> np.ndarray:
    return reader(msg.quaternion)


@reader.register
def _(msg: geometry_msgs.msg.Pose2D) -> np.ndarray:
    return np.array([msg.x, msg.y, msg.theta], dtype=np.float32)


@reader.register
def _(msg: geometry_msgs.msg.Pose) -> np.ndarray:
    return np.concatenate([reader(msg.position), reader(msg.orientation)])


@reader.register
def _(msg: geometry_msgs.msg.PoseStamped) -> np.ndarray:
    return reader(msg.pose)


@reader.register
def _(msg: geometry_msgs.msg.PoseWithCovariance) -> np.ndarray:
    return reader(msg.pose)


@reader.register
def _(msg: geometry_msgs.msg.PoseWithCovarianceStamped) -> np.ndarray:
    return reader(msg.pose)


@reader.register
def _(msg: geometry_msgs.msg.PoseArray) -> Optional[np.ndarray]:
    return reader(msg.poses)


@reader.register
def _(msg: geometry_msgs.msg.Twist) -> np.ndarray:
    return np.concatenate([reader(msg.linear), reader(msg.angular)])


@reader.register
def _(msg: geometry_msgs.msg.TwistStamped) -> np.ndarray:
    return reader(msg.twist)


@reader.register
def _(msg: geometry_msgs.msg.TwistWithCovariance) -> np.ndarray:
    return reader(msg.twist)


@reader.register
def _(msg: geometry_msgs.msg.TwistWithCovarianceStamped) -> np.ndarray:
    return reader(msg.twist)


@reader.register
def _(msg: geometry_msgs.msg.Transform) -> np.ndarray:
    return np.concatenate([reader(msg.translation), reader(msg.rotation)])


@reader.register
def _(msg: geometry_msgs.msg.TransformStamped) -> np.ndarray:
    return reader(msg.transform)


@reader.register
def _(msg: sensor_msgs.msg.JointState) -> np.ndarray:
    return np.array(msg.position, dtype=np.float32)


@reader.register
def _(msg: sensor_msgs.msg.LaserScan) -> np.ndarray:
    result = np.array(msg.ranges, dtype=np.float32)
    result[result < msg.range_min] = np.nan
    result[result >= msg.range_max] = msg.range_max
    return result


@reader.register
def _(msg: sensor_msgs.msg.Imu) -> np.ndarray:
    return np.concatenate([reader(msg.orientation), reader(msg.angular_velocity), reader(msg.linear_acceleration)])


@reader.register
def _(msg: nav_msgs.msg.Odometry) -> np.ndarray:
    return np.concatenate([reader(msg.pose), reader(msg.twist)])


@reader.register
def _(msg: nav_msgs.msg.Path) -> Optional[np.ndarray]:
    return reader(msg.poses)


@reader.register
def _(msg: visualization_msgs.msg.Marker) -> np.ndarray:
    return reader(msg.pose)


@reader.register
def _(msg: visualization_msgs.msg.MarkerArray) -> Optional[np.ndarray]:
    return reader(msg.markers)


def lookup_transform_from_buffer(frame: str, wrt: str, buffer: tf2_ros.Buffer) -> np.ndarray:
    try:
        time = buffer.get_latest_common_time(wrt, frame)
    except tf2_ros.LookupException:
        return np.full((7,), np.nan)
    except tf2_ros.TransformException:
        return np.full((7,), np.nan)

    if not buffer.can_transform(target_frame=wrt, source_frame=frame, time=time):
        return np.full((7,), np.nan)

    tf = buffer.lookup_transform(target_frame=wrt, source_frame=frame, time=time)
    return reader(tf)


_buffer = tf2_ros.Buffer(rclpy.duration.Duration(seconds=1e9))


@reader.register
def _(msg: tf2_msgs.msg.TFMessage, is_static: bool = False, buffer: tf2_ros.Buffer = _buffer) -> Optional[np.ndarray]:
    if not msg.transforms:
        return None

    for tf in msg.transforms:
        if is_static:
            buffer.set_transform_static(tf, "bagfile")
        else:
            buffer.set_transform(tf, "bagfile")

    return np.stack(
        [lookup_transform_from_buffer(frame=f, wrt=wrt, buffer=buffer) for f, wrt in frame_to_save_wrt.items()],
        axis=-1,
    )


@reader.register
def _(msg: azure_kinect_ros_msgs.msg.MarkerArrayStamped) -> Optional[np.ndarray]:
    if not msg.markers:
        return np.array([[np.nan]])

    return reader([m.pose for m in msg.markers])


######################


def interpolate(target_times: np.ndarray, times: np.ndarray, data: np.ndarray, kind: str = "zero") -> np.ndarray:
    f = scipy.interpolate.interp1d(times, data, kind=kind, axis=0, assume_sorted=True, fill_value="extrapolate")
    return f(target_times)


def sanitize_topics(bag: BagReader, topics: List[str], fn) -> List[str]:
    result = []
    for topic in topics:
        if topic not in bag.type_map:
            bag.logger.warning(f'Cannot import from "{topic}", not present in the bag')
            continue

        msg_type = bag.get_message_type(topic)
        if topic in bag.type_map and hasattr(fn, "registry") and msg_type in fn.registry:
            result.append(topic)
        else:
            bag.logger.warning(f'Cannot import from "{topic}" of type "{msg_type}"')

    def move_to_front(e: Any, l: List[Any]) -> None:
        if e in l:
            l.remove(e)
            l.insert(0, e)

    move_to_front("/tf", result)
    move_to_front("/tf_static", result)

    return result


def import_topic(
    bag: BagReader, topic: str, use_header_stamps: bool, sync: bool, sync_stamps: Optional[np.ndarray] = None
) -> Optional[Dict[str, np.ndarray]]:
    datas = []
    stamps = []

    for _, msg, stamp in bag.get_messages(topics=[topic]):
        try:
            if topic == "/tf_static":
                data = reader(msg, is_static=True)
            else:
                data = reader(msg)

            if data is not None:
                datas.append(data)
        except ReaderException:
            msg_type = bag.get_message_type(topic)
            bag.logger.warning(f'Cannot import from "{topic}" of type "{msg_type}"')
            return None

        if use_header_stamps:
            try:
                stamp = header_stamp(msg)
            except AttributeError:
                pass

        if data is not None:
            stamps.append(stamp)

    if not datas:
        msg_type = bag.get_message_type(topic)
        bag.logger.warning(f'Cannot import from "{topic}" of type "{msg_type}"')
        return None

    shapes = np.array(list(map(lambda x: x.shape, datas)))
    min_shape = shapes.min(axis=0)
    max_shape = shapes.max(axis=0)

    # note: when msgs have different shape pad with zeros to match max_shape
    if not (min_shape == max_shape).all():
        padded_datas = np.full([len(datas)] + max_shape.tolist(), fill_value=np.nan)
        for i, d in enumerate(datas):
            index = (i,) + tuple([slice(0, s) for s in d.shape])
            padded_datas[index] = d
        datas = padded_datas

    if sync and sync_stamps is not None:
        datas = interpolate(sync_stamps, np.array(stamps), np.array(datas))

    result = dict(messages=datas)

    if sync and sync_stamps is None:
        result["stamps"] = stamps

    return result


def export_bag(
    file: pathlib.Path,
    output_file: str,
    output_folder: str,
    topics: List[str] = [],
    metadata: List[str] = [],
    use_header_stamps: bool = True,
    sync_topic: Optional[str] = None,
) -> None:
    sync_stamps = None
    bag = BagReader(str(file))
    sync = sync_topic is not None

    output_folder = output_folder if output_folder != "" else file.parent

    hdf5_file_name = f"{output_file}.h5" if output_file != "" else f"{file.stem}.h5"
    hdf5_file_path = os.path.join(output_folder, hdf5_file_name)
    # Check if file already exists
    if os.path.exists(hdf5_file_path):
        raise FileExistsError(f"Output file '{hdf5_file_path}' already exists please delete it or choose another name.")

    store = h5py.File(hdf5_file_path, "w")

    clean_topics = sanitize_topics(bag, topics, reader)
    clean_metadata = sanitize_topics(bag, metadata, metadata_reader)

    if sync:
        if sync_topic not in clean_topics:
            raise ValueError(f'Sync topic "{sync_topic}" cannot be read.')

        clean_topics.remove(sync_topic)
        topic_data = import_topic(
            bag=bag,
            topic=sync_topic,
            use_header_stamps=use_header_stamps,
            sync=sync,
            sync_stamps=None,
        )

        sync_stamps = topic_data["stamps"]
        store.create_dataset("stamps", data=topic_data["stamps"])
        store.create_dataset(f"{sanitize(sync_topic)}", data=topic_data["messages"])

        bag.logger.info(f'Imported "{sync_topic}"')

    for topic in clean_metadata:
        bag.logger.info(f'Will try to import metadata from "{topic}"')
        _, msg, _ = next(bag.get_messages(topics=[topic]))

        try:
            topic_metadata = metadata_reader(msg)
        except ReaderException:
            topic_metadata = None
            msg_type = bag.get_message_type(topic)
            bag.logger.warning(f'Cannot import metadata from "{topic}" of type "{msg_type}"')

        if topic_metadata is not None:
            for k, v in topic_metadata.items():
                store.attrs[f"{sanitize(topic)}_{k}"] = v

            bag.logger.info(f'Imported metadata from "{topic}"')

    for topic in clean_topics:
        bag.logger.info(f'Will try to import messages from "{topic}"')

        topic_data = import_topic(
            bag=bag,
            topic=topic,
            use_header_stamps=use_header_stamps,
            sync=sync,
            sync_stamps=sync_stamps,
        )

        if topic_data is not None:
            store.create_dataset(f"{sanitize(topic)}", data=topic_data["messages"])
            bag.logger.info(f'Imported messges from "{topic}"')

            if topic == "/body_tracking_data":
                num_body_markers = 32
                body_count = (~np.isnan(topic_data["messages"][..., 0])).sum(axis=-1) // num_body_markers
                store.create_dataset(f"body_count", data=body_count)

            if topic == "/tf":
                for i, t in enumerate(frame_to_save_wrt.keys()):
                    store.create_dataset(f"tf_{t}", data=store["tf"][..., i])

    store.close()


def export_bags(
    file: pathlib.Path,
    output_folder: str,
    topics: List[str] = [],
    metadata: List[str] = [],
    use_header_stamps: bool = True,
    sync_topic: Optional[str] = None,
) -> None:
    """
    Reads a file containing paths of bag files and processes each one sequentially.

    Args:
        file (pathlib.Path): Path to the file containing a list of bag file paths.
        output_folder (str): The folder where the output files will be stored.
        topics (List[str]): List of topics to export.
        metadata (List[str]): List of topics to extract metadata from.
        use_header_stamps (bool): Whether to use header stamps for synchronization.
        sync_topic (Optional[str]): Topic to synchronize data on.
    """
    # Read the file containing the paths to bag files
    with open(file, "r") as f:
        bag_files = [pathlib.Path(line.strip()) for line in f if line.strip()]

    if not bag_files:
        raise ValueError("The file does not contain any valid bag file paths.")

    for bag_file in bag_files:
        try:
            # Determine the output file name based on the current bag file
            output_file_name = bag_file.parent.name

            # Process the bag file using the export_bag function
            export_bag(
                file=bag_file,
                output_file=output_file_name,
                output_folder=output_folder,
                topics=topics,
                metadata=metadata,
                use_header_stamps=use_header_stamps,
                sync_topic=sync_topic,
            )
        except Exception as e:
            print(f"Failed to process bag file '{bag_file}': {e}")


def main(args: Any = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bags_info_file", help="File containing a list of bag files paths", type=pathlib.Path)
    parser.add_argument("--output_folder", help="Output folder", default="")
    parser.add_argument("--topics", help="topics", type=str, nargs="+", default=topics_names)
    parser.add_argument("--exclude", help="exclude topics", type=str, nargs="+", default="")
    parser.add_argument("--use_header_stamps", help="use stamps from headers", type=bool, default=True)
    parser.add_argument(
        "--sync_on_topic", help="whether to sync on a given topic", type=str, default=sync_dataset_on_topic
    )

    # Parse arguments
    arg = parser.parse_args(args) if args else parser.parse_args()

    export_bags(
        file=arg.bags_info_file,
        output_folder=arg.output_folder,
        topics=topics_names,
        metadata=save_metadata_of_topics,
        sync_topic=arg.sync_on_topic,
        use_header_stamps=arg.use_header_stamps,
    )
