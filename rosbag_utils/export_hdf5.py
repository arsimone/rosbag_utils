import argparse
import os
from typing import Any, Collection

import h5py

try:
    from robomaster_msgs.msg import H264Packet
except ImportError:
    H264Packet = None

try:
    from h264_msgs.msg import Packet
except:
    Packet = None


from .reader import BagReader, header_stamp, sanitize


def import_topic(bag: BagReader, topic: str, msg_type: Any, store: h5py.File,
                 use_header_stamps: bool = True
                 ) -> bool:
    if not hasattr(msg_type, 'reader'):
        bag.logger.warning(f'Cannot import messages of type {msg_type}')
        return False
    datas = []
    stamps = []
    for _, msg, stamp in bag.get_messages(topics=[topic]):
        try:
            data = msg.reader()
            if data is not None:
                datas.append(data)
        except Exception as e:  # noqa
            bag.logger.warning(f'Cannot import message: {e}')
            return False
        if use_header_stamps:
            try:
                stamp = header_stamp(msg)
            except AttributeError:
                pass
        stamps.append(stamp)
    if datas:
        try:
            store.create_dataset(f"{sanitize(topic)}:data", data=datas)
        except ValueError:
            bag.logger.warning(f'Failed to import topic {topic}')
            return False
    if stamps:
        store.create_dataset(f"{sanitize(topic)}:stamp", data=stamps)
    return len(datas) > 0 or len(stamps) > 0


def import_h264_stamps(bag: BagReader, topic: str, topic_type: Any,
                       store: h5py.File, use_header_stamps: bool = True
                       ) -> bool:
    from .h264_video import h264_stamps

    stamps = h264_stamps(bag, topic, use_header_stamps)
    if stamps:
        store.create_dataset(f"{sanitize(topic)}:stamp", data=stamps)
    return len(stamps) > 0


def export_bag(bag_file: str, topics: Collection[str] = [], exclude: Collection[str] = [],
               use_header_stamps: bool = True, should_make_video: bool = False,
               video_format: str = 'mp4') -> None:
    bag = BagReader(bag_file)
    bag_name = os.path.basename(os.path.normpath(bag_file))
    if not topics:
        topics = bag.type_map.keys()
    if exclude:
        topics = set(topics) - set(exclude)
    with h5py.File(f'{bag_name}.h5', 'w') as store:
        for topic in topics:
            bag.logger.info(f'Will try to import {topic}')
            msg_type = bag.get_message_type(topic)
            if msg_type is None:
                continue
            if msg_type in (H264Packet, Packet):
                t = import_h264_stamps(bag, topic, msg_type, store, use_header_stamps)
                if should_make_video:

                    from .h264_video import make_video

                    out = f'{bag_name}__{sanitize(topic)}.{video_format}'
                    make_video(bag, topic, out)
            else:
                t = import_topic(bag, topic, msg_type, store, use_header_stamps)
            if t:
                bag.logger.info(f'imported {topic}')


def main(args: Any = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('bag_file', help='Bag file')
    parser.add_argument('--topics', help='topics', type=str, nargs='+', default="")
    parser.add_argument('--exclude', help='exclude topics', type=str, nargs='+', default="")
    parser.add_argument('--use_header_stamps', help='use stamps from headers', type=bool,
                        default=True)
    parser.add_argument('--make_video', help='make video', type=bool, default=False)
    parser.add_argument('--video_format', help='video format', type=str, default='mp4')
    arg = parser.parse_args(args)
    export_bag(arg.bag_file, arg.topics, arg.exclude, arg.use_header_stamps, arg.make_video,
               arg.video_format)
