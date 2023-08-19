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

@reader(users_landmarks_msgs.msg.InteractionUsersData)
def interaction_users_data(msg: users_landmarks_msgs.msg.InteractionUsersData) -> Dict:
    interaction_users_data = {}
    interaction_users_data["multiple_users_landmarks_list"] = multiple_users_landmarks(msg.users_data)
    interaction_users_data["looking_at_camera_list"] = [label for label in msg.looking_at_camera]
    interaction_users_data["sequence_id"] = msg.sequence_id
    return interaction_users_data

@reader(users_landmarks_msgs.msg.InteractionSequenceLabel)
def interaction_sequence_label(msg: users_landmarks_msgs.msg.InteractionSequenceLabel) -> Dict:
    interaction_sequence_label = {}
    interaction_sequence_label["timestamp"] = msg.header.stamp.sec + msg.header.stamp.nanosec/1000000000
    interaction_sequence_label["interacted"] = msg.interacted
    interaction_sequence_label["sequence_id"] = msg.sequence_id
    return interaction_sequence_label

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
               topics: Collection[str] = [],
               exclude: Collection[str] = [],
               use_header_stamps: bool = True,
               labels_to_add: list = None) -> None:
    bag = BagReader(bag_file)
            
    if not topics:
        topics = bag.type_map.keys()
    if exclude:
        topics = set(topics) - set(exclude)
    
    print(f'Will try to import topics: {topics}')
    
    topics_dict = {}
            
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
            topic_list = []
            
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
                    
                    topic_list.append(user_dict)
            topics_dict[topic] = topic_list
            
        elif msg_type == users_landmarks_msgs.msg.InteractionUsersData:
            print(f"Number of messages: {len(topic_messages_list)}")
            topic_list = []
            
            for user_idx in range(len(topic_messages_list)):
                sequence_id = topic_messages_list[user_idx]["sequence_id"]
                users_labels = topic_messages_list[user_idx]["looking_at_camera_list"]
                users = topic_messages_list[user_idx]["multiple_users_landmarks_list"]
                for user in users:
                    user_dict = {}
                    user_dict["timestamp"] = user["timestamp"]
                    user_dict["frame_id"] = user["frame_id"]
                    user_dict["body_id"] = user["body_id"]
                    user_dict["looking_at_camera"] = users_labels[0]
                    user_dict["sequence_id"] = sequence_id
                    
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
                    
                    topic_list.append(user_dict)    
            topics_dict[topic] = topic_list
        
        elif msg_type == users_landmarks_msgs.msg.InteractionSequenceLabel:
            print(f"Number of messages: {len(topic_messages_list)}")
            topic_list = []
            for sequence_idx in range(len(topic_messages_list)):
                topic_list.append(topic_messages_list[sequence_idx])
            topics_dict[topic] = topic_list
        
    return topics_dict
    
def export_bags(bags_info_file_path: str,
                output_folder: str = None,
                topics: Collection[str] = [],
                exclude: Collection[str] = [],
                use_header_stamps: bool = True) -> None:
    
    print("Exporting bags specified in:", bags_info_file_path)

    with open(bags_info_file_path) as bags_info_file:
        bags_info_dict = yaml.load(bags_info_file, Loader=yaml.FullLoader)
    
    bags_topics_lists_dict = {}
    
    for bag_info in bags_info_dict["bags_info"]:
        print(f'Extracting bag file: {bag_info["path"]}')

        labels_to_add_names = [key_name for key_name in bag_info.keys()]
        labels_to_add_names.remove("path")

        labels_to_add = [{"label_name": label_name, "label_value": bag_info[label_name]} for label_name in labels_to_add_names]
        
        bag_topics_lists_dict = export_bag(bag_info["path"], topics, exclude, use_header_stamps, labels_to_add)
        
        # Check if topics already are in bags_topics_lists_dict
        for topic_name in bag_topics_lists_dict.keys():
            if topic_name in bags_topics_lists_dict.keys():
                bags_topics_lists_dict[topic_name].extend(bag_topics_lists_dict[topic_name])
            else:
                bags_topics_lists_dict[topic_name] = bag_topics_lists_dict[topic_name]

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    for topic_name in bags_topics_lists_dict.keys():
        output_file = os.path.join(output_folder, f'{topic_name[1:]}.csv')
        print(f'Exporting specified bag files content to {output_file}')
        dataframe = pd.DataFrame.from_dict(bags_topics_lists_dict[topic_name])
        print(dataframe.info())
        dataframe.to_csv(output_file, index=False, header=True)                            
                
def main(args: Any = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bags_info_file', help='File containing a list of bag files')
    parser.add_argument('--dataset_name', help='Name for dataset')
    parser.add_argument('--topics', help='topics', type=str, nargs='+', default="")
    parser.add_argument('--exclude', help='exclude topics', type=str, nargs='+', default="")
    parser.add_argument('--use_header_stamps', help='use stamps from headers', type=bool,
                        default=True)
    parser.add_argument('--make_video', help='make video', type=bool, default=False)
    parser.add_argument('--video_format', help='video format', type=str, default='mp4')
    arg = parser.parse_args(args)
    
    if arg.bags_info_file is None:
        raise ValueError('Must specify bags_info_file')
    
    output_csv_files_folder = os.path.join(os.path.dirname(arg.bags_info_file), "csv_files")
    
    if arg.bags_info_file is not None:
        export_bags(arg.bags_info_file, output_csv_files_folder, arg.topics, arg.exclude, arg.use_header_stamps)
