# How to import a bag file in Python
# see rosbag2/rosbag2_py/test/test_sequential_reader.py

import argparse
from datetime import datetime
from typing import Tuple, Iterator, Any, List, Type, Callable, Optional

import numpy as np
import scipy.interpolate

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
import rclpy.logging

Message = Any


def reader(t: Any) -> Callable[[Any], Callable[[Any], Optional[np.ndarray]]]:
    def g(f: Callable[[Any], np.ndarray]) -> Callable[[Any], Optional[np.ndarray]]:
        setattr(t, 'reader', f)
        return f
    return g


def interpolate(target_times: np.ndarray, times: np.ndarray, data: np.ndarray,
                kind: str = 'nearest') -> np.ndarray:
    print(f"E {times.shape} {data.shape}")
    f = scipy.interpolate.interp1d(times, data, kind=kind, axis=0, assume_sorted=True,
                                   fill_value="extrapolate")
    return f(target_times)


def get_rosbag_options(path: str, serialization_format: str = 'cdr'
                       ) -> Tuple[rosbag2_py.StorageOptions, rosbag2_py.ConverterOptions]:
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)
    return storage_options, converter_options


def message_type_is_known(msg: str) -> bool:
    try:
        get_message(msg)
        return True
    except AttributeError:
        return False


def header_stamp(msg: Any) -> int:
    stamp = msg.header.stamp
    return int(1e9 * stamp.sec + stamp.nanosec)


def sanitize(name: str, replacement: str = "__") -> str:
    if name[0] == '/':
        name = name[1:]
    return name.replace('/', replacement)


class BagReader:

    def __init__(self, bag_path: str):
        self.logger = rclpy.logging.get_logger("rosbag utils")
        self.storage_options, self.converter_options = get_rosbag_options(bag_path)
        reader = rosbag2_py.SequentialReader()
        reader.open(self.storage_options, self.converter_options)
        topic_types = reader.get_all_topics_and_types()
        self.type_map = {topic_types[i].name: topic_types[i].type for i in range(len(topic_types))}
        self.count = {t: 0 for t in self.type_map}
        self.initial_stamp = 0
        self.final_stamp = 0
        while reader.has_next():
            (topic, data, self.final_stamp) = reader.read_next()
            self.count[topic] += 1
            if self.initial_stamp == 0:
                self.initial_stamp = self.final_stamp

    def __repr__(self) -> str:
        t0 = datetime.fromtimestamp(self.initial_stamp / 1e9)
        t1 = datetime.fromtimestamp(self.final_stamp / 1e9)
        self.final_stamp // 1e9
        max_name = max(len(n) for n in self.type_map)
        max_type = max(len(n) for n in self.type_map.values())
        # max_number = max(len(str(n)) for n in self.count.values())
        topics = '\n'.join(f"{n.ljust(max_name)} | {v.ljust(max_type)} [{'x' if message_type_is_known(v) else '-'}] | {self.count[n]}"
                           for n, v in self.type_map.items())
        return f"""
First message: {t0:%Y-%m-%d %H:%M:%S.%f}
Last message: {t1:%Y-%m-%d %H:%M:%S.%f}
Topics:
{topics}
"""

    def get_messages(self, topics: List[str] = []) -> Iterator[Tuple[str, Message, int]]:
        # Set filter for topic of string type
        storage_filter = rosbag2_py.StorageFilter(topics=topics)
        reader = rosbag2_py.SequentialReader()
        reader.open(self.storage_options, self.converter_options)
        reader.set_filter(storage_filter)
        while reader.has_next():
            (topic, data, t) = reader.read_next()
            msg_type = self.get_message_type(topic)
            try:
                msg = deserialize_message(data, msg_type)
            except (TypeError, Exception) as e:
                self.logger.error(f"Cannot deserialize message of type {msg_type}: {e}")
                continue
            yield (topic, msg, t)

    def get_message_type(self, topic: str) -> Type:
        return get_message(self.type_map[topic])

    def import_topic(self, topic: str, use_header_stamps: bool = True
                     ) -> Tuple[List[Any], List[int]]:
        msg_type = self.get_message_type(topic)
        if msg_type is None or not hasattr(msg_type, 'reader'):
            self.logger.warning(f'Cannot import messages of topic {topic}')
            return ([], [])
        datas = []
        stamps = []
        for _, msg, stamp in self.get_messages(topics=[topic]):
            try:
                data = msg.reader()
                if data is not None:
                    datas.append(data)
                else:
                    continue
            except Exception as e:  # noqa
                self.logger.warning(f'Cannot import message: {e}')
                return ([], [])
            if use_header_stamps:
                try:
                    stamp = header_stamp(msg)
                except AttributeError:
                    pass
            stamps.append(stamp)
        return datas, stamps


def main(args: Any = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('bag_file', help='Bag file')
    arg = parser.parse_args(args)
    bag = BagReader(arg.bag_file)
    print(bag)
