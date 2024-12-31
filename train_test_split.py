import os
import shutil
import random


source_folder = "BraTS2021_Training_Data"
validation_folder = "BraTS2021_Validation_Data"
test_folder = "BraTS2021_Test_Data"


os.makedirs(validation_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)


subfolders = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]

random.shuffle(subfolders)


num_validation = int(len(subfolders) * 0.1)
num_test = int(len(subfolders) * 0.2)


validation_subfolders = subfolders[:num_validation]
test_subfolders = subfolders[num_validation:num_validation + num_test]

for folder in validation_subfolders:
    src_path = os.path.join(source_folder, folder)
    dst_path = os.path.join(validation_folder, folder)
    shutil.move(src_path, dst_path)

for folder in test_subfolders:
    src_path = os.path.join(source_folder, folder)
    dst_path = os.path.join(test_folder, folder)
    shutil.move(src_path, dst_path)

print(f"Moved {len(validation_subfolders)} subfolders to {validation_folder}.")
print(f"Moved {len(test_subfolders)} subfolders to {test_folder}.")
