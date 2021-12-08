import os,sys
import numpy as np
from collections import defaultdict

# !pip install opencv-python
import cv2
# from google.colab.patches import cv2_imshow

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

def load_data(file_path, test_n=1):  
  # load numpy file
  data = np.load(file_path, allow_pickle=True)

  # split into img, label
  # data = np.array(data)
  images = np.array(list(data[:, 0]), dtype=np.float)
  labels = np.array(list(data[:, 1]), dtype=np.int)
  # labels = np.argmax(labels, axis=1)
  # print(labels[0])
  
  return data, images, labels

HOTKEY_DICT = {0: 'W_VEC', 1: 'S_VEC', 2: 'A_VEC', 3: 'D_VEC', 4: 'WA_VEC', 
              5: 'WD_VEC', 6: 'SA_VEC', 7: 'SD_VEC', 8: 'NK_VEC'}
HOTKEY_DICT_RES = {v:k for k,v in HOTKEY_DICT.items()}
def get_lbl_str(labels):
  return [HOTKEY_DICT[label.argmax()] for label in labels]

def get_toAugment_data(imgs, labels):
  # Count each label
  from collections import Counter
  label_strs = get_lbl_str(labels)
  label_counts_ori = dict(Counter(label_strs))
  print('label_counts_ori', label_counts_ori)

  threshold = img_n//3
  aumentation_labels = [label for label, count in label_counts_ori.items() if count < threshold]
  print('aumentation_labels', aumentation_labels )

  # get data to be augmented
  img_nonAugmented_dict, img_toAugmented_dict = defaultdict(list), defaultdict(list)
  for img, label_onehot in zip(imgs, labels):
    label = get_lbl_str([label_onehot])[0]
    if label in aumentation_labels:
      img_toAugmented_dict[label].append(img)
    else:
      img_nonAugmented_dict[label].append(img)

  return img_nonAugmented_dict, img_toAugmented_dict

# Data Augmentation
def get_tf_ds(imgs, label):
  labels = [label]*len(imgs)
  ds = tf.data.Dataset.from_tensor_slices((imgs, labels))
  return ds, imgs, labels

def augmentate(imgs, label, ways):
  ds, imgs, labels = get_tf_ds(img_toAugmented_dict[label], label)

  dataset = None
  for i, (way, params) in enumerate(ways.items()):
    if not isinstance(params, list): params = [params]
    for param in params:
      dataset_new  = ds.map(lambda x, label: color(x, way, param), num_parallel_calls=4)
      dataset = dataset.concatenate(dataset_new) if dataset else dataset_new

  return ds, dataset

def show(ds_ori, dataset):
  plt.figure(figsize=(50, 50))
  plt.tight_layout()

  # init for all aumentation operations & imgs
  operations = ['hue', 'saturation', 'contrast']
  processed_imgs = {operation:[] for operation in operations}

  # store original img
  ori_imgs = [img for i, (img, label) in enumerate(ds_ori)] 
  processed_imgs['original'] = ori_imgs
  img_n = len(ori_imgs)

  # store augmented imgs
  operation_n = 0  
  for i, image in enumerate(dataset):
    operation = operations[operation_n]
    processed_imgs[operation].append(image)
    if (i+1)%img_n == 0 and (i+1)>=img_n:
      operation_n += 1
  # print({k:len(v) for k,v in processed_imgs.items()})

  # plot imgs
  plt.figure(figsize=(50, 50))
  col = len(processed_imgs)
  img_i = 1
  for operation_i, operation in enumerate(['original', 'hue', 'saturation', 'contrast']):
    img = processed_imgs[operation][img_i]

    plt.subplot(1, col, operation_i+1)
    plt.imshow(img.numpy().astype("uint8"))
    plt.title(operation)

  plt.show()

def df2np(ds, label_str):
  label_onehot = [0]*9
  label_onehot[HOTKEY_DICT_RES[label_str]] = 1
  
  return np.array([[img, label_onehot] for img in ds])
  # return np.array([[np.array(img), label_onehot] for img in ds])

def save_np(data_np, output_dir, filename_ori):
  filename_new = '{}_augmented_color.npy'.format(filename_ori.replace('.npy',''))
  output_path = os.path.join(output_dir, filename_new)

  with open(output_path,'wb') as f:
    np.save(f, data_np)
  
  print(f'Save the augmented data as {output_path}\n')


# x: tf.Tensor, way: str, param:str
def color(x, way, param=None) -> tf.Tensor:
  if way == 'hue':
    if not param:
      return tf.image.random_hue(x, 0.08, 1.02)
    else:
      return tf.image.adjust_hue(x, param)
  elif way == 'saturation':
    if not param:
      return tf.image.random_saturation(x, 0.2, 0.8)
    else:
      return tf.image.adjust_saturation(x, param)
  elif way == 'brightness':
    if not param:
      return tf.image.random_brightness(x, 0.3, 0.7)
    else:
      return tf.image.adjust_brightness(x, param)
  elif way == 'contrast':
    if not param:
      return tf.image.random_contrast(x, 0.7, 1.3)
    else:
      return tf.image.adjust_contrast(x, param)

def main():
  # reading parameters
  in_path = sys.argv[1]
  output_dir = in_path if len(sys.argv) < 2 else argv[2]
  test_n = int(sys.argv[3]) if len(sys.argv)==3 else -1

  WAYS = {'hue': [0.1, 0.5], 'saturation': 1.5, 'contrast': 0.5}

  # load input paths
  file_paths = [os.path.join(in_path, f) for f in next(os.walk(in_path))[2] if f.endswith('.npy')]
  if test_n != -1:
    file_paths = file_paths[:test_n]
      
  # process each image dataset
  for file_path_i, file_path in enumerate(file_paths):
    print(f'Processing the {file_path_i+1}th file ...')
    data, imgs, labels = load_data(file_path)
    img_n = len(imgs)

    # get nonAugmented/toAugmented image dataset
    img_nonAugmented_dict, img_toAugmented_dict = get_toAugment_data(imgs, labels)
    # img_toAugmented_dict = get_toAugment_data(imgs, labels)
    # print({k:len(v) for k, v in img_toAugmented_dict.items()})

    # augment data by labels
    augmented_data = []
    for label in img_toAugmented_dict.keys():
      ds_ori, ds_augmented = augmentate(img_toAugmented_dict[label], label, WAYS)
      # if label=='S_VEC': show(ds_ori, ds_augmented)
      
      # convert tf.dataset to numpy[img, label_onehot]
      augmented_np = df2np(ds_augmented, label)
      augmented_data.extend(augmented_np)

      print(f'{label}: {len(img_toAugmented_dict[label])} -> {len(augmented_np)}')

    # combine all data
    all_data = augmented_data
    for label, ds_ori in img_nonAugmented_dict.items():
      all_data.extend(df2np(ds_ori, label))

    # save as numpy format
    filename_ori =  file_path.split('/')[-1]
    save_np(augmented_data, output_dir, filename_ori)


if __name__ == '__main__':
  main()