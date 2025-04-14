import os
import pickle
import numpy as np
from PIL import Image

class HeadCameraDataProcessor:
    '''
    Pre-processing .pkl data:
    1. Extract head_camera data from .pkl files.
    2. Sample 20 .pkl files from each episode using Gaussian.
    3. Resize the RGB images to 128x128 and normalize them.
    4. Save the processed data into `extracted_resized_head_camera_data.pkl`.
    '''
    
    def __init__(self, subdataset_dirs, num_samples=20, output_file='extracted_resized_head_camera_data.pkl'):
        self.subdataset_dirs = subdataset_dirs
        self.num_samples = num_samples
        self.output_file = output_file

    def extract_head_camera_data(self, file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data['observation']['head_camera']

    def sample_pkl_files(self, episode_dir):
        pkl_files = [f for f in os.listdir(episode_dir) if f.endswith('.pkl')]
        pkl_files.sort()
        total_files = len(pkl_files)
        idx = np.arange(total_files)
        weights = np.exp(-0.5 * ((idx - total_files // 2) / (total_files // 4))**2)
        weights /= weights.sum()
        sampled_files = np.random.choice(pkl_files, size=self.num_samples, p=weights, replace=False)
        head_camera_data = []
        for file in sampled_files:
            file_path = os.path.join(episode_dir, file)
            head_camera_data.append(self.extract_head_camera_data(file_path))
        return head_camera_data

    def resize_and_normalize_image(self, image_array):
        image = Image.fromarray(image_array)
        image = image.resize((128, 128))
        image_array = np.array(image)
        return image_array / 255.0

    def process_subdataset(self, subdataset_dir):
        all_data = {}
        for episode_num in range(100):
            episode_dir = os.path.join(subdataset_dir, f"episode{episode_num}")
            if os.path.exists(episode_dir):
                head_camera_data = self.sample_pkl_files(episode_dir)
                resized_data = []
                for data in head_camera_data:
                    resized_data.append({
                        'intrinsic_cv': data['intrinsic_cv'],
                        'extrinsic_cv': data['extrinsic_cv'],
                        'cam2world_gl': data['cam2world_gl'],
                        'rgb': self.resize_and_normalize_image(data['rgb'])
                    })
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



def process_data():
    subdataset_dirs = [
        'datasets/block_hammer_beat_D435_pkl',
        'datasets/block_handover_D435_pkl',
        'datasets/blocks_stack_easy_D435_pkl'
    ]
    
    processor = HeadCameraDataProcessor(subdataset_dirs=subdataset_dirs, num_samples=20)
    processor.process_all_subdatasets()


if __name__ == "__main__":
    process_data()






# The following shows print out the structure of the processed data 
# ============================================================================

# import pickle
# def load_and_explore_pkl(file_path):
#     with open(file_path, 'rb') as file:
#         data = pickle.load(file)
    
#     print(f"Loaded data from {file_path}")
#     print("Data structure overview:")
    
#     for subdataset in data:
#         print(f"\n{subdataset}: {len(data[subdataset])} episodes")
#         sample_episode = list(data[subdataset].keys())[0]
#         sample_head_camera_data = data[subdataset][sample_episode]
#         print(f"Number of samples in the first episode: {len(sample_head_camera_data)}")
        
#         if sample_head_camera_data:
#             first_sample = sample_head_camera_data[0]
#             print(f"Keys in the 'head_camera' data: {list(first_sample.keys())}")
#             if 'rgb' in first_sample:
#                 print(f"Shape of 'rgb' array in the first sample: {first_sample['rgb'].shape}")
    
#     return data

# file_path = 'extracted_resized_head_camera_data.pkl'
# data = load_and_explore_pkl(file_path)

# ---------------------------------------------------------------------------------

# Loaded data from extracted_resized_head_camera_data.pkl
# Data structure overview:

# block_hammer_beat_D435_pkl: 100 episodes
# Number of samples in the first episode: 10
# Keys in the 'head_camera' data: ['intrinsic_cv', 'extrinsic_cv', 'cam2world_gl', 'rgb']
# Shape of 'rgb' array in the first sample: (128, 128, 3)

# block_handover_D435_pkl: 100 episodes
# Number of samples in the first episode: 10
# Keys in the 'head_camera' data: ['intrinsic_cv', 'extrinsic_cv', 'cam2world_gl', 'rgb']
# Shape of 'rgb' array in the first sample: (128, 128, 3)

# blocks_stack_easy_D435_pkl: 100 episodes
# Number of samples in the first episode: 10
# Keys in the 'head_camera' data: ['intrinsic_cv', 'extrinsic_cv', 'cam2world_gl', 'rgb']
# Shape of 'rgb' array in the first sample: (128, 128, 3)
