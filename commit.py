'''
Author: Aurora
Date: 2025-05-04 21:54:01
Descripttion: 
LastEditTime: 2025-05-04 21:55:42
'''
split_dataset = ip2p_dataset.train_test_split(test_size=0.2)

train_dataset = split_dataset['train']
test_dataset = split_dataset['test']


REPO_ID = 'Aurora1609/RoboTwin'

# 将训练集推送到 Hugging Face Hub
train_dataset.push_to_hub(
    repo_id=REPO_ID,
    split='train',
    private=False
)
print(f"训练集已成功推送到 {REPO_ID} 仓库的训练集分割中。")

# 将测试集推送到 Hugging Face Hub
test_dataset.push_to_hub(
    repo_id=REPO_ID,
    split='test',
    private=False
)
print(f"测试集已成功推送到 {REPO_ID} 仓库的测试集分割中。")
    