import os
from shutil import copyfile, copytree
import time
import re

# copies the params and result files (csv/json etc) AND the last checkpoint folder
copy_from_dir = "./data/HalfCheetahBulletEnv-v0/PPO/"
copy_to_dir = "./Results/HalfCheetah/PPO/FCN_2/"

for subdir, dirs, files in os.walk(copy_from_dir):
    for dir in dirs:
        for subdir2, dirs2, files2 in os.walk(subdir + dir):
            if not os.path.exists(copy_to_dir + dir):
                os.makedirs(copy_to_dir + dir)
            for file2 in files2:
                copyfile(subdir + dir + "/" + file2, copy_to_dir + dir + "/" + file2)
            max_folder = 0
            for folder in dirs2:
                if int(re.match('checkpoint_(.*)', folder).group(1)) > max_folder:
                    max_folder = int(re.match('checkpoint_(.*)', folder).group(1))
            copytree(subdir + dir + "/" + "checkpoint_" + str(max_folder), copy_to_dir + dir + "/" + "checkpoint_" + str(max_folder))
            print("copied!")
            break
    break
