import argparse
import os
from typing import Any, Dict, Callable, Optional, Collection

import numpy as np
import pandas as pd
import yaml

import nav_msgs.msg
import geometry_msgs.msg
import sensor_msgs.msg
import users_landmarks_msgs.msg 
from users_landmarks_utils.kinect_bt_utils import body_joints_list, body_joints_info
import transforms3d

from .reader import BagReader, header_stamp, sanitize

_readers: Dict[Any, Callable[[Any], np.ndarray]] = {}


def reader(t: Any) -> Callable[[Any], Callable[[Any], Optional[np.ndarray]]]:
    def g(f: Callable[[Any], np.ndarray]) -> Callable[[Any], Optional[np.ndarray]]:
        setattr(t, 'reader', f)
        return f
    return g

@reader(users_landmarks_msgs.msg.SingleUserLandmarks)
def single_users_landmarks(msg: users_landmarks_msgs.msg.SingleUserLandmarks) -> Dict:
    single_users_landmarks = {}
    
    single_users_landmarks["timestamp"] = msg.header.stamp.sec + msg.header.stamp.nanosec/1000000000
    
    single_users_landmarks["frame_id"] = msg.header.frame_id
    
    single_users_landmarks["body_id"] = msg.body_id
    
    single_users_landmarks["body_landmarks"] = [get_pose_data(pose) for pose in msg.body_landmarks]
    
    single_users_landmarks["face_landmarks"] = [np.asarray([landmark.x, landmark.y, landmark.z], dtype=np.float32) for landmark in msg.face_landmarks]  
    
    return single_users_landmarks

@reader(users_landmarks_msgs.msg.MultipleUsersLandmarks)
def multiple_users_landmarks(msg: users_landmarks_msgs.msg.MultipleUsersLandmarks) -> list:
    multiple_users_landmarks = []
    for single_user_landmarks in msg.users:
        multiple_users_landmarks.append(single_users_landmarks(single_user_landmarks))    
    
    return multiple_users_landmarks

def get_pose_data(pose):
    pose_data = {}
    pose_data["position"] = np.asarray([pose.position.x,
                                           pose.position.y,
                                           pose.position.z], dtype=np.float32)
    pose_data["orientation"] = np.asarray([pose.orientation.x, 
                                          pose.orientation.y, 
                                          pose.orientation.z,
                                          pose.orientation.w], dtype=np.float32)
    return pose_data

def import_topic(bag: BagReader, topic: str, 
                 msg_type: Any, 
                 use_header_stamps: bool = True) -> bool:
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
        return datas, stamps
    else:
        return None


def export_bag(bag_file: str,
               output_file: str = None,
               write_to_file: bool = False,
               topics: Collection[str] = [],
               exclude: Collection[str] = [],
               use_header_stamps: bool = True,
               labels_to_add: list = None) -> None:
    bag = BagReader(bag_file)
    bag_name = os.path.basename(os.path.normpath(bag_file))
            
    if not topics:
        topics = bag.type_map.keys()
    if exclude:
        topics = set(topics) - set(exclude)
        
    dictionary_list = []
            
    for topic in topics:
        bag.logger.info(f'Will try to import {topic}')
        msg_type = bag.get_message_type(topic)
        topic_messages_list, topic_headers_list = import_topic(bag, topic, msg_type, use_header_stamps)
        if topic_messages_list is None:
            bag.logger.info(f'Could not import topic: {topic}')
            continue
        
        # Got list of messages
        bag.logger.info(f'imported {topic}')
        if msg_type == users_landmarks_msgs.msg.MultipleUsersLandmarks:
            print(f"Number of messages: {len(topic_messages_list)}")
            
            for users in topic_messages_list:
                for user in users:
                    user_dict = {}
                    user_dict["timestamp"] = user["timestamp"]
                    user_dict["frame_id"] = user["frame_id"]
                    user_dict["body_id"] = user["body_id"]
                    
                    for body_landmark_idx, body_landmark in enumerate(user['body_landmarks']):
                        # Position
                        field_name = f'body_joint_{body_joints_list[body_landmark_idx]}_position_x'
                        user_dict[field_name] = body_landmark["position"][0]
                        field_name = f'body_joint_{body_joints_list[body_landmark_idx]}_position_y'
                        user_dict[field_name] = body_landmark["position"][1]
                        field_name = f'body_joint_{body_joints_list[body_landmark_idx]}_position_z'
                        user_dict[field_name] = body_landmark["position"][2]
                        # Orientation
                        field_name = f'body_joint_{body_joints_list[body_landmark_idx]}_quaternion_x'
                        user_dict[field_name] = body_landmark["orientation"][0]
                        field_name = f'body_joint_{body_joints_list[body_landmark_idx]}_quaternion_y'
                        user_dict[field_name] = body_landmark["orientation"][1]
                        field_name = f'body_joint_{body_joints_list[body_landmark_idx]}_quaternion_z'
                        user_dict[field_name] = body_landmark["orientation"][2]
                        field_name = f'body_joint_{body_joints_list[body_landmark_idx]}_quaternion_w'
                        user_dict[field_name] = body_landmark["orientation"][3]
                        
                    
                    for face_landmark_idx, face_landmark in enumerate(user['face_landmarks']):
                        field_name = f'face_landmark_{face_landmark_idx}_image_x'
                        user_dict[field_name] = face_landmark[0]
                        field_name = f'face_landmark_{face_landmark_idx}_image_y'
                        user_dict[field_name] = face_landmark[1]
                        field_name = f'face_landmark_{face_landmark_idx}_image_z'
                        user_dict[field_name] = face_landmark[2]
                    
                    if labels_to_add is not None:
                        for label in labels_to_add:
                            user_dict[label["label_name"]] = label["label_value"]
                    
                    dictionary_list.append(user_dict)
    
    if write_to_file:     
        output_file_path = os.path.splitext(bag_name)[0] + '.csv'           
        if output_file is None:
            output_file_path = os.path.join(os.path.dirname(bag_file), output_file)
        else:
            if not os.path.isdir(os.path.dirname(output_file)):
                os.mkdir(os.path.dirname(output_file))
            output_file_path = output_file            
            
        print(f'Exporting: \n{bag_file} \nto \n{output_file}')
        
        dataframe = pd.DataFrame.from_dict(dictionary_list)
        dataframe.to_csv(output_file, index=False, header=True)
    else:
        return dictionary_list
    
def export_bags(bags_info_file_path: str,
                output_file: str = None,
                topics: Collection[str] = [],
                exclude: Collection[str] = [],
                use_header_stamps: bool = True) -> None:
    
    

    with open(bags_info_file_path) as bags_info_file:
        bags_info_dict = yaml.load(bags_info_file, Loader=yaml.FullLoader)
    
    bags_dictionary_list = []
    
    for bag_info in bags_info_dict["bags_info"]:
        print(f'Extracting bag file: {bag_info["path"]}')

        labels_to_add_names = [key_name for key_name in bag_info.keys()]
        labels_to_add_names.remove("path")

        labels_to_add = [{"label_name": label_name, "label_value": bag_info[label_name]} for label_name in labels_to_add_names]
        bags_dictionary_list.extend(export_bag(bag_info["path"], None, False, topics, exclude, use_header_stamps, labels_to_add))
    
    if not os.path.isdir(os.path.dirname(output_file)):
        os.mkdir(os.path.dirname(output_file))
                            
    print(f'Exporting specified bag files content to {output_file}')
        
    dataframe = pd.DataFrame.from_dict(bags_dictionary_list)
    print(dataframe.info())
    dataframe.to_csv(output_file, index=False, header=True)                            
                
def main(args: Any = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_file', help='Bag file')
    parser.add_argument('--bags_info_file', help='File containing a list of bag files')
    parser.add_argument('--output_file', help='Output file name for CSV')
    parser.add_argument('--topics', help='topics', type=str, nargs='+', default="")
    parser.add_argument('--exclude', help='exclude topics', type=str, nargs='+', default="")
    parser.add_argument('--use_header_stamps', help='use stamps from headers', type=bool,
                        default=True)
    parser.add_argument('--make_video', help='make video', type=bool, default=False)
    parser.add_argument('--video_format', help='video format', type=str, default='mp4')
    arg = parser.parse_args(args)
    
    if arg.bag_file and arg.bags_info_file:
        raise ValueError('Cannot specify both bag_file and bags_info_file')
    if arg.bag_file is None and arg.bags_info_file is None:
        raise ValueError('Must specify either bag_file or bags_info_file')
    
    if arg.bag_file is not None:
        export_bag(arg.bag_file, arg.output_file, True, arg.topics, arg.exclude, arg.use_header_stamps, None)
    else:
        export_bags(arg.bags_info_file, arg.output_file, arg.topics, arg.exclude, arg.use_header_stamps)
