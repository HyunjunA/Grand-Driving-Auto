import time
import cv2
import numpy as np
from grabscreen import grab_screen
from getkeys import key_check
import sys
import os
from constants import IMAGE_WIDTH,IMAGE_HEIGHT,NUM_KEYS,W_VEC,A_VEC,S_VEC,D_VEC,WA_VEC,WD_VEC,SA_VEC,SD_VEC,NK_VEC


def key_label(keys) :

    output = [0,0,0,0,0,0,0,0,0]

    if 'W' in keys and 'A' in keys:
        output = WA_VEC
    elif 'W' in keys and 'D' in keys:
        output = WD_VEC
    elif 'S' in keys and 'A' in keys:
        output = SA_VEC
    elif 'S' in keys and 'D' in keys:
        output = SD_VEC
    elif 'W' in keys:
        output = W_VEC
    elif 'S' in keys:
        output = S_VEC
    elif 'A' in keys:
        output = A_VEC
    elif 'D' in keys:
        output = D_VEC
    else:
        output = NK_VEC
    return output


def save_data(output_file,data):

    if not os.path.exists(output_file):
        print("Saving data to new file")
    with open(output_file,'wb'):
        np.save(output_file,data)


def main():

    output_dir = sys.argv[1]
    if len(sys.argv) > 2:
        file_num = int(sys.argv[2])
    else :
        curr_files = [file.split('-')[-1].split('.')[0] for root,dirs,files in os.walk(output_dir,topdown=False) for file in files]
        file_nums = []
        for file_val in curr_files :
            try :
                file_nums.append(int(file_val))
            except:
                continue
        if not file_nums :
            file_num = 1
        else :
            file_num = max(file_nums) + 1
    frame_count = 0
    training_data = []
    paused = False
    print("Starting in")
    for i in list(range(5))[::-1] :
        print(i)
        time.sleep(1)

    while True:
        start = time.time()
        if not paused:
            img = grab_screen((0,0,800,600))
            img = cv2.resize(img,(IMAGE_WIDTH,IMAGE_HEIGHT))
            keys = key_check()
            key_output = key_label(keys)
            training_data.append([img,key_output])
            frame_count += 1
            if len(training_data) % 100 == 0:
                print(f"{frame_count} Total Frames Collected")
                if len(training_data) % 500 == 0:
                    print('Saving data')
                    output_file = os.path.join(output_dir,f"training_data-{file_num}.npy")
                    save_data(output_file,training_data)
                    training_data = []
                    file_num += 1
            # cv2.imshow("frame", img)
            end = time.time()
            #print(f"Frame took {end-start} seconds")
            cv2.waitKey(100)
            if cv2.waitKey(1) & 0Xff == ord('q'):
                break

        keys = key_check()
        if 'P' in keys:
            if paused:
                paused = False
                print('Unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)

    cv2.destroyAllWindows()

if __name__ == '__main__' :

    main()