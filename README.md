# Grand Driving Auto

## Data Collection:
Must have game running in a 800x600 window in the top left corner of the screen

    python get_data.py <output_dir>

Data will be saved in the output dir, in files called training_data-{file_num}.npy
if files are already present in the output dir, the new data will be stored at the next file_num increment.
500 images per training data file
Data collection functions referenced from below articles: 

https://pythonprogramming.net/direct-input-game-python-plays-gta-v/?completed=/open-cv-basics-python-plays-gta-v/

https://pythonprogramming.net/game-frames-open-cv-python-plays-gta-v/


## Balance Data
Augment data to fix imbalance problem with color adjust.
- Save augmented file as `xxx_augmented_color.npy` in the same directory of inputted data.
```
python balance_image.py {INPUT_DIR}
```

- Specify output directory
```
python balance_image.py {INPUT_DIR} {OUTPUT_DIR}
```

- Specify amount of augmented file
```
python balance_image.py {INPUT_DIR} {OUTPUT_DIR} {AUGMENTED_N}
```

## Train A Model:

### V1

    python train_model.py -d <data dir> -n <num files of data> [-m] <model name> [-e] <num epochs> [-b] <batch size> [-lr] <learning_rate>

### V2 (using static validation set)

    python train_model_v2.py -d <training data dir> -dv <validation data dir> -nt <num files of train data> -nv <num files of val data> [-m] <model name> [-e] <num epochs> [-b] <batch size> [-lr] <learning_rate> 

Model Names: CNN, AlexNet, AlexNetV2, Xception, ResNet50 or InceptionV3

### V3 (include LSTM models, switch to fit generator instead of train on batch)

    python train_model_v3.py -d <training data dir> -dv <validation data dir> -nt <num files of train data> -nv <num files of val data> [-m] <model name> [-e] <num epochs> [-b] <batch size> [-s] <sequence length> [-lr] <learning_rate> 

add flag -l to load a model from a checkpoint file

    Example: python train_model_v3.py -d ./data -dv ./data -nt 1 -nv 1 -m InceptionV3 -e 1 -b 1 -s 10 -lr 0.01 
    
Model Names: ConvLSTM, LSTM, CNN, AlexNet, AlexNetV2, Xception, ResNet50 or InceptionV3


## Model Testing

### Test CNN Models
Must have game running in a 800x600 window in the top left corner of the screen
    
    python test_model.py <absolute_path_to_saved_model>

### Test LSTM Models

    python test_model_lstm.py <seq_len> <model_name> <absolute_path_to_saved_LSTM_model>

Model names: ObjectDetection or ConvLSTM

## GPU or CPU
If you want to use GPU, please remove this below. The code is the first part of the main function.
```
    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
```
## Tensorboard
tensorboard dev upload --logdir .\tb_logs
