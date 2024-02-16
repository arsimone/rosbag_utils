import argparse
import os
from typing import Any, Dict, Callable, Optional, Collection

import numpy as np
import pandas as pd
import yaml

import std_msgs.msg
import classifiers_output_msgs.msg 
import robomaster_user_reaction_msgs.msg

from .reader import BagReader, header_stamp, sanitize

_readers: Dict[Any, Callable[[Any], np.ndarray]] = {}

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

def reader(t: Any) -> Callable[[Any], Callable[[Any], Optional[np.ndarray]]]:
    def g(f: Callable[[Any], np.ndarray]) -> Callable[[Any], Optional[np.ndarray]]:
        setattr(t, 'reader', f)
        return f
    return g

@reader(std_msgs.msg.String)
def string(msg: std_msgs.msg.String) -> Dict:
    return {"timestamp" : msg.header.stamp.sec + msg.header.stamp.nanosec/1000000000, "data" : msg.data}

@reader(classifiers_output_msgs.msg.MultiUserBinaryClassifierOutput)
def multi_user_binary_classifier_output(msg: classifiers_output_msgs.msg.MultiUserBinaryClassifierOutput) -> Dict:
    multi_user_binary_classifier_output = {}
    multi_user_binary_classifier_output["timestamp"] = msg.header.stamp.sec + msg.header.stamp.nanosec/1000000000
    multi_user_binary_classifier_output["user_ids"] = [user_id for user_id in msg.user_ids]
    multi_user_binary_classifier_output["probabilities"] = [probability for probability in msg.probabilities]
    multi_user_binary_classifier_output["users_torso_poses"] = [get_pose_data(pose) for pose in msg.users_torso_poses]
    return multi_user_binary_classifier_output

@reader(robomaster_user_reaction_msgs.msg.OfferingDemoStateEvent)
def offering_demo_state_event(msg: robomaster_user_reaction_msgs.msg.OfferingDemoStateEvent) -> Dict:
    offering_demo_state_event = {}
    offering_demo_state_event["timestamp"] = msg.header.stamp.sec + msg.header.stamp.nanosec/1000000000
    offering_demo_state_event["event_type"] = msg.event_type
    offering_demo_state_event["user_id"] = msg.user_id
    offering_demo_state_event["probability"] = msg.probability
    offering_demo_state_event["distance"] = msg.distance
    return offering_demo_state_event

@reader(robomaster_user_reaction_msgs.msg.OfferingDemoModeRequest)
def offering_demo_mode_request(msg: robomaster_user_reaction_msgs.msg.OfferingDemoModeRequest) -> Dict:
    return {"timestamp" : msg.header.stamp.sec + msg.header.stamp.nanosec/1000000000, "requested_mode" : msg.requested_mode}

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
        if msg_type == classifiers_output_msgs.msg.MultiUserBinaryClassifierOutput and topic == "/iid_obt/model_output":
            print(f"Number of messages: {len(topic_messages_list)}")
            topic_list = []
            topic_prefix = "iid_obt_model_output"
            for model_output in topic_messages_list:
                for user_idx in range(len(model_output["user_ids"])):
                    model_output_dict = {f"{topic_prefix}_timestamp": model_output["timestamp"],
                                         f"{topic_prefix}_user_ids": model_output["user_ids"][user_idx],
                                         f"{topic_prefix}_probabilities": model_output["probabilities"][user_idx],
                                         f"{topic_prefix}_torso_pos_x": model_output["users_torso_poses"][user_idx]["position"][0],
                                         f"{topic_prefix}_torso_pos_y": model_output["users_torso_poses"][user_idx]["position"][1],
                                         f"{topic_prefix}_torso_pos_z": model_output["users_torso_poses"][user_idx]["position"][2],
                                         f"{topic_prefix}_torso_quat_x": model_output["users_torso_poses"][user_idx]["orientation"][0],
                                         f"{topic_prefix}_torso_quat_y": model_output["users_torso_poses"][user_idx]["orientation"][1],
                                         f"{topic_prefix}_torso_quat_z": model_output["users_torso_poses"][user_idx]["orientation"][2],
                                         f"{topic_prefix}_torso_quat_w": model_output["users_torso_poses"][user_idx]["orientation"][3]}
                    topic_list.append(model_output_dict)
                
            topics_dict[topic] = topic_list
            
        elif msg_type == classifiers_output_msgs.msg.MultiUserBinaryClassifierOutput and topic == "/iid_mg/model_output":
            print(f"Number of messages: {len(topic_messages_list)}")
            topic_list = []
            topic_prefix = "iid_mg_model_output"
            for model_output in topic_messages_list:
                for user_idx in range(len(model_output["user_ids"])):
                    model_output_dict = {f"{topic_prefix}_timestamp": model_output["timestamp"],
                                         f"{topic_prefix}_user_ids": model_output["user_ids"][user_idx],
                                         f"{topic_prefix}_probabilities": model_output["probabilities"][user_idx],
                                         f"{topic_prefix}_torso_pos_x": model_output["users_torso_poses"][user_idx]["position"][0],
                                         f"{topic_prefix}_torso_pos_y": model_output["users_torso_poses"][user_idx]["position"][1],
                                         f"{topic_prefix}_torso_pos_z": model_output["users_torso_poses"][user_idx]["position"][2],
                                         f"{topic_prefix}_torso_quat_x": model_output["users_torso_poses"][user_idx]["orientation"][0],
                                         f"{topic_prefix}_torso_quat_y": model_output["users_torso_poses"][user_idx]["orientation"][1],
                                         f"{topic_prefix}_torso_quat_z": model_output["users_torso_poses"][user_idx]["orientation"][2],
                                         f"{topic_prefix}_torso_quat_w": model_output["users_torso_poses"][user_idx]["orientation"][3]}
                    topic_list.append(model_output_dict)
                
            topics_dict[topic] = topic_list
        
        elif msg_type == robomaster_user_reaction_msgs.msg.OfferingDemoStateEvent and topic == "/rm_user_offering_node/state_event":
            print(f"Number of messages: {len(topic_messages_list)}")
            topic_list = []
            topic_prefix = "offering_event"
            for offering_event in topic_messages_list:
                offering_event_dict = {f"{topic_prefix}_timestamp": offering_event["timestamp"],
                                       f"{topic_prefix}_event_type": offering_event["event_type"],
                                       f"{topic_prefix}_user_id": offering_event["user_id"],
                                       f"{topic_prefix}_probability": offering_event["probability"],
                                       f"{topic_prefix}_distance": offering_event["distance"]}
                topic_list.append(offering_event_dict)
                
            topics_dict[topic] = topic_list    
        
        elif msg_type == robomaster_user_reaction_msgs.msg.OfferingDemoModeRequest and topic == "/rm_user_offering_node/requested_mode":
            print(f"Number of messages: {len(topic_messages_list)}")
            topic_list = []
            topic_prefix = "mode_request"
            for mode_request in topic_messages_list:
                mode_dict = {f"{topic_prefix}_timestamp": mode_request["timestamp"],
                             f"{topic_prefix}_mode_requested": mode_request["requested_mode"]}
                topic_list.append(mode_dict)                
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
        output_file = os.path.join(output_folder, f'{topic_name[1:].replace("/", "_")}.csv')
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
    print (f"Output folder: {output_csv_files_folder}")
    
    if arg.bags_info_file is not None:
        export_bags(arg.bags_info_file, output_csv_files_folder, arg.topics, arg.exclude, arg.use_header_stamps)
