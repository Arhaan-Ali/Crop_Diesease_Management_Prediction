import os
import random
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

source_dir = os.path.join(BASE_DIR, "dataset", "blast")
train_dir = os.path.join(BASE_DIR, "finalised_dataset", "Train", "tungro")
test_dir = os.path.join(BASE_DIR, "finalised_dataset", "Test", "tungro")
val_dir = os.path.join(BASE_DIR, "finalised_dataset", "Validate", "tungro")

# create folders if they don’t exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

images = [entry.name for entry in os.scandir(source_dir) if entry.is_file()]

Total = len(images)
Test = int(Total * 0.15)
Validate = Test
Train = Total - (Test + Validate)

random.shuffle(images)

# TEST
for i in range(0, Test):
    src = os.path.join(source_dir, images[i])
    dst = os.path.join(test_dir, images[i])
    shutil.copy(src, dst)

# VALIDATE
for i in range(Test, Test + Validate):
    src = os.path.join(source_dir, images[i])
    dst = os.path.join(val_dir, images[i])
    shutil.copy(src, dst)

# TRAIN
for i in range(Test + Validate, Total):
    src = os.path.join(source_dir, images[i])
    dst = os.path.join(train_dir, images[i])
    shutil.copy(src, dst)

print("Dataset split completed successfully.")