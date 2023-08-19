import argparse
from typing import Any, Dict, Callable, Optional, Collection
import os
import yaml

def find_files_with_extension(folder_path, extension):
    file_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                file_list.append(os.path.join(root, file))
    return file_list

def write_yaml_file(file_list, output_file):
    data = {}
    data['bags_info'] = [{'path': file, 'experiment_number': (file_id+1)} for file_id, file in enumerate(file_list)]
    with open(output_file, 'w') as yaml_file:
        yaml.dump(data, yaml_file)

def main(args: Any = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', help='folder_path')
    arg = parser.parse_args(args)
    
    folder_path = arg.folder_path
    extension = ".db3"
    output_file_path = os.path.join(arg.folder_path, "dataset_description.yaml")  # Replace with the desired output YAML file name

    file_list = sorted(find_files_with_extension(folder_path, extension))
    
    print(f"Found {len(file_list)} files with extension '{extension}' in folder '{folder_path}':")
    for file in file_list:
        print(file)
    
    write_yaml_file(file_list, output_file_path)
    
    print(f"YAML file '{output_file_path}' created with {len(file_list)} files.")