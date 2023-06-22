import argparse
from typing import Any, Iterator, List
import tempfile
import os

import numpy as np
import libmedia_codec

from .reader import BagReader, header_stamp, sanitize


def h264_frames(video_decoder: libmedia_codec.H264Decoder, data: bytes) -> Iterator[np.ndarray]:
    frames = video_decoder.decode(data)
    for frame_data in frames:
        (frame, width, height, ls) = frame_data
        if frame:
            frame = np.frombuffer(frame, dtype=np.ubyte, count=len(frame))
            yield frame.reshape((height, width, 3))


def h264_stamps(bag: BagReader, topic: str, use_header_stamps: bool = True) -> List[int]:
    video_decoder = libmedia_codec.H264Decoder()
    stamps: List[int] = []
    for _, msg, stamp in bag.get_messages(topics=[topic]):
        frames = h264_frames(video_decoder, msg.data.tobytes())
        if use_header_stamps:
            stamp = header_stamp(msg)
        for frame in frames:
            stamps.append(stamp)
    return stamps


def h264_buffer(bag: BagReader, topic: str) -> np.ndarray:
    return np.concatenate(
        [np.frombuffer(frame.data, np.uint8)
         for _, frame, _ in bag.get_messages(topics=[topic])])


def make_video(bag: BagReader, topic: str, out: str) -> None:
    # read the whole data
    data = h264_buffer(bag, topic)
    with tempfile.NamedTemporaryFile() as f:
        f.write(data.tobytes())
        os.system(f'ffmpeg -i {f.name} -vcodec copy {out}')


def main(args: Any = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('bag_file', help='Bag file')
    parser.add_argument('topic', help='Topic name')
    parser.add_argument('--out', help='video out file', default=".mp4")
    arg = parser.parse_args(args)
    bag_name = os.path.basename(os.path.normpath(arg.bag_file))
    out = f'{bag_name}__{sanitize(arg.topic)}{arg.out}'
    bag = BagReader(arg.bag_file)
    make_video(bag, arg.topic, out)
