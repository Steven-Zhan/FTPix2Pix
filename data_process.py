from datasets import load_dataset
from huggingface_hub import login
from datasets import Dataset
import json
import os
import pickle
import numpy as np
from PIL import Image
import json


class HeadCameraDataProcessor:
    '''
    Pre-processing .pkl data:
    1. Extract head_camera data from .pkl files.
    2. Sample 20 .pkl files from each episode using Gaussian.
    3. Resize the RGB images to 128x128 and normalize them.
    4. Save the processed data into `extracted_resized_head_camera_data.pkl`.
    5. Convert the processed data to JSON format and write to file with specific format.
    '''

    def __init__(self, subdataset_dirs, num_samples=20, output_file='extracted_resized_head_camera_data.pkl',
                 json_output_file='output2.json'):
        self.subdataset_dirs = subdataset_dirs
        self.num_samples = num_samples
        self.output_file = output_file
        self.json_output_file = json_output_file

    def extract_head_camera_data(self, file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data['observation']['head_camera']

    def sample_pkl_files(self, episode_dir):
        pkl_files = [f for f in os.listdir(episode_dir) if f.endswith('.pkl')]
        pkl_files.sort()
        # 只考虑第一个到倒数第二个文件
        available_files = pkl_files[:-1]
        total_available_files = len(available_files)
        # if total_available_files < self.num_samples:
        #     raise ValueError(f"可用文件数量 {total_available_files} 少于要采样的数量 {self.num_samples}")
        idx = np.arange(total_available_files)
        weights = np.exp(-0.5 * ((idx - total_available_files // 2) / (total_available_files // 4)) ** 2)
        weights /= weights.sum()
        sampled_files = np.random.choice(available_files, size=self.num_samples, p=weights, replace=False)
        before = []
        after = []
        for file in sampled_files:
            file_path = os.path.join(episode_dir, file)
            before.append(self.extract_head_camera_data(file_path))
            current_index = pkl_files.index(file)
            next_file = pkl_files[current_index + 1]
            next_file_path = os.path.join(episode_dir, next_file)
            after.append(self.extract_head_camera_data(next_file_path))
        return before, after

    def resize_and_normalize_image(self, image_array):
        image = Image.fromarray(image_array)
        image = image.resize((128, 128))
        image_array = np.array(image)
        return np.round(image_array / 255.0, 4)

    def process_subdataset(self, subdataset_dir):
        all_data = {}
        for episode_num in range(100):
            episode_dir = os.path.join(subdataset_dir, f"episode{episode_num}")
            if os.path.exists(episode_dir):
                before_data, after_data = self.sample_pkl_files(episode_dir)
                resized_data = []
                for i in range(len(before_data)):
                    resized_before = {
                        'intrinsic_cv': before_data[i]['intrinsic_cv'],
                        'extrinsic_cv': before_data[i]['extrinsic_cv'],
                        'cam2world_gl': before_data[i]['cam2world_gl'],
                        'rgb': self.resize_and_normalize_image(before_data[i]['rgb'])
                    }
                    resized_after = {
                        'intrinsic_cv': after_data[i]['intrinsic_cv'],
                        'extrinsic_cv': after_data[i]['extrinsic_cv'],
                        'cam2world_gl': after_data[i]['cam2world_gl'],
                        'rgb': self.resize_and_normalize_image(after_data[i]['rgb'])
                    }
                    resized_data.append((resized_before, resized_after))
                all_data[f"episode{episode_num}"] = resized_data
        return all_data

    def process_all_subdatasets(self):
        all_subdataset_data = {}
        for subdataset_dir in self.subdataset_dirs:
            subdataset_name = os.path.basename(subdataset_dir)
            print(f"Processing {subdataset_name}...")
            subdataset_data = self.process_subdataset(subdataset_dir)
            all_subdataset_data[subdataset_name] = subdataset_data
        with open(self.output_file, 'wb') as output_file:
            pickle.dump(all_subdataset_data, output_file)
        print(f"Data processed and saved to {self.output_file}")

    def convert_to_json(self):
        file_path = self.output_file
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        print("load over")
        json_data_list = []
        for subdataset in data:
            for episode in data[subdataset]:
                episode_data = data[subdataset][episode]
                for before_data, after_data in episode_data:
                    original_images = before_data['rgb'].tolist()
                    edited_images = after_data['rgb'].tolist()


                    # prompt
                    if str(subdataset) == 'block_hammer_beat_D435_pkl':
                        prompts = "Currently, a robot is using a hammer to strike a block placed on a table. " \
                        "Predict what the camera will capture in the scene 50 frames into the future. " \
                        "Consider factors such as the motion dynamics of the robot, the physical interactions between the hammer, " \
                        "the block, and the table, and any potential changes in the spatial arrangement of objects."
                    elif str(subdataset) == 'block_handover_D435_pkl':
                        prompts = "Currently, the robot is using arm movement to transfer a block to a handover point. " \
                        "Based on the present scene, predict what the camera will capture in the scene 50 frames later, " \
                        "considering the typical steps of the block hand - over process " \
                        "such as the transfer between arms and the movement towards the target."
                    elif str(subdataset) == 'blocks_stack_easy_D435_pkl':
                        prompts = "Currently, the robot is in the process of placing the black block and red block. " \
                        "Based on the present scene, predict what the camera will capture in the scene 30 frames later. " \
                        "Consider the relative positions of the two blocks, the movement speed of the robot's arm while handling the blocks," \
                        "and any possible adjustments during the placement."
                    else:
                        print('nonono here is problem')
                    json_data = {
                        "before": original_images,
                        "after": edited_images,
                        "prompt": prompts
                    }
                    json_data_list.append(json_data)

        # 重新组织数据为符合 Dataset.from_dict 的格式
        new_data = {
            "before": [obj["before"] for obj in json_data_list],
            "after": [obj["after"] for obj in json_data_list],
            "prompt": [obj["prompt"] for obj in json_data_list]
        }

        with open(self.json_output_file, 'w') as json_file:
            json.dump(new_data, json_file, indent=4)

        print("Data converted to JSON and saved to", self.json_output_file)


def process_data():
    subdataset_dirs = [
        'datasets/block_hammer_beat_D435_pkl',
        'datasets/block_handover_D435_pkl',
        'datasets/blocks_stack_easy_D435_pkl'
    ]

    processor = HeadCameraDataProcessor(subdataset_dirs=subdataset_dirs, num_samples=20)
    processor.process_all_subdatasets()
    processor.convert_to_json()

    # 加载并推送数据集
    login(token='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')
    with open('output1.json', 'r') as f:
        data = json.load(f)
    ip2p_dataset = Dataset.from_dict(data)
    REPO_ID = 'Aurora1609/Pix2Pix_RoboTwin'
    ip2p_dataset.push_to_hub(
        repo_id=REPO_ID,
        split='train',
        private=False
    )
    print(f"数据集已成功推送到 {REPO_ID} 仓库的训练集分割中。")


if __name__ == "__main__":
    process_data()