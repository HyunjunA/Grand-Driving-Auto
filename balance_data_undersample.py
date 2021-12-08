import os
import numpy as np
from constants import W_VEC,A_VEC,S_VEC,D_VEC,WA_VEC,WD_VEC,SA_VEC,SD_VEC,NK_VEC
from collections import Counter



input_dir = "F:/validation_data"
output_dir = 'F:/validation_data_balanced'

labels = []
i = 0
for root, dirs, files in os.walk(input_dir):
    for file in files:
        file_path = os.path.join(root,file)
        data = np.load(file_path,allow_pickle=True)
        labels.extend(data[:,1])
        #if i>5:
            #break
        i += 1

label_count = {'W':0,'A':0,'S':0,'D':0,'WA':0,'WD':0,'SA':0,'SD':0,'NK':0}

for label in labels:
    if label == W_VEC:
        label_count['W'] += 1
    elif label == A_VEC:
        label_count['A'] += 1
    elif label == S_VEC:
        label_count['S'] += 1
    elif label == D_VEC:
        label_count['D'] += 1
    elif label == WA_VEC:
        label_count['WA'] += 1
    elif label == WD_VEC:
        label_count['WD'] += 1
    elif label == SA_VEC:
        label_count['SA'] += 1
    elif label == SD_VEC:
        label_count['SD'] += 1
    elif label == NK_VEC:
        label_count['NK'] += 1

print(label_count)
print(min(label_count,key=label_count.get))

