import sys
import numpy as np
import os
import os.path
from os import path

'''
This is a one time use script written to fix the file names of the training data
DO NOT RUN this script again!!!!
'''
data_dir = sys.argv[1]

def change_names():
    for file_num in range(1, 62):
        file_path = os.path.join(data_dir, f"training_data-{file_num}.npy")
        file_data = np.load(file_path, allow_pickle=True)
        new_file_num = file_num + 141
        export_file = os.path.join(data_dir,f"training_data-{new_file_num}.npy")
        save_data(export_file, file_data)
    
def save_data(output_file,data):
    with open(output_file,'wb'):
        np.save(output_file,data)

change_names()