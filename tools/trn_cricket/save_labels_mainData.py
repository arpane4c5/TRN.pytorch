#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 11:25:28 2019

@author: arpan

@Description: Generate feature vector labels and save to target folder

"""

import numpy as np
import os
import json


# Local Paths
LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
MAIN_DATASET = "/home/arpan/VisionWorkspace/Cricket/dataset_25_fps_train_set"
MAIN_LABELS = "/home/arpan/VisionWorkspace/Cricket/dataset_25_fps_train_set_labels"
VAL_DATASET = "/home/arpan/VisionWorkspace/Cricket/dataset_25_fps_val_set"
VAL_LABELS = "/home/arpan/VisionWorkspace/Cricket/dataset_25_fps_val_set_labels"
TEST_DATASET = "/home/arpan/VisionWorkspace/Cricket/dataset_25_fps_test_set"
TEST_LABELS = "/home/arpan/VisionWorkspace/Cricket/dataset_25_fps_test_set_labels"

# Server Paths
if os.path.exists("/opt/datasets/cricket/ICC_WT20"):
    LABELS = "/home/arpan/VisionWorkspace/shot_detection/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/opt/datasets/cricket/ICC_WT20"
    MAIN_DATASET = "/home/arpan/DATA_Drive/Cricket/dataset_25_fps_train_set"
    MAIN_LABELS = "/home/arpan/DATA_Drive/Cricket/dataset_25_fps_train_set_labels"
    VAL_DATASET = "/home/arpan/DATA_Drive/Cricket/dataset_25_fps_val_set"
    VAL_LABELS = "/home/arpan/DATA_Drive/Cricket/dataset_25_fps_val_set_labels"
    TEST_DATASET = "/home/arpan/DATA_Drive/Cricket/dataset_25_fps_test_set"
    TEST_LABELS = "/home/arpan/DATA_Drive/Cricket/dataset_25_fps_test_set_labels"

BASEPATH = "../../cricket_highlights"
#C3D_FEATS_TRAIN_DIR = "c3dFinetunedOnHLMainSeq23_mainDataset_train_feats_17"
#C3D_FEATS_VAL_DIR = "c3dFinetunedOnHLMainSeq23_mainDataset_val_feats_17"
#C3D_FEATS_TEST_DIR = "c3dFinetunedOnHLMainSeq23_mainDataset_test_feats_17"
TARGET_DIR = "target"


# Split the dataset files into training, validation and test sets
# All video files present at the same path (assumed)
def split_dataset_files(datasetPath):
    filenames = sorted(os.listdir(datasetPath))         # read the filename
    filenames = [t.split('.')[0] for t in filenames]   # remove the extension
    return filenames[:16], filenames[16:21], filenames[21:]


# All video files present at the same path (assumed)
def get_main_dataset_files(datasetPath):
    vfiles = sorted(os.listdir(datasetPath))         # read the filename
    return vfiles

# return the number of frames present in a video vid
def getNFrames(vid):
    import cv2
    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        import sys
        print("Capture Object not opened ! Abort")
        sys.exit(0)
        
    l = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return l

def write_target(video_prefix, video_path, labels_path):
    
    nClasses = 2
    
    labfile = os.path.join(labels_path, video_prefix+'.json')
    assert os.path.isfile(labfile), "File does not exist : {}".format(labfile)
    assert os.path.isdir(os.path.join(BASEPATH, TARGET_DIR)), "Target path does not exist!"
        
    with open(labfile, 'r') as fobj:
        shots = json.load(fobj)

    # get number of frames in the video
    video_file = os.path.join(video_path, video_prefix+".avi")
    if not os.path.isfile(video_file):
        video_file = os.path.join(video_path, video_prefix+'.mp4')
    video_size = getNFrames(video_file)
        
    # Check seq_size is valid ie., less than min action size
    ## TODO:
    
    target = np.zeros((video_size, nClasses))
    
    k = list(shots.keys())[0]  #taking the values 
    pos = shots[k]  # get list of tuples for kn
    
    for (start, end) in pos:
        target[start:end+1, 1] = 1
        
    target[target[:,1]==0, 0] = 1
    
    np.save(os.path.join(BASEPATH, TARGET_DIR, video_prefix+".npy"), target)
        
    
def write_data_info(class_index, train_list, test_list):
    
    cricket_dict = {"CRICKET": {"class_index": class_index,  \
                    "train_session_set": train_list, \
                    "test_session_set": test_list}}
    
    with open(os.path.join(BASEPATH, "../data", "data_info.json"), "w") as fp:
        json.dump(cricket_dict, fp)
    


if __name__ == '__main__':
    
    assert os.path.isdir(BASEPATH), "{} does not exist!".format(BASEPATH)
    
    # Form dataloaders 
    train_lst_main_ext = get_main_dataset_files(MAIN_DATASET)   #with extensions
    train_lst_main = [t.rsplit('.', 1)[0] for t in train_lst_main_ext]   # remove the extension
    val_lst_main_ext = get_main_dataset_files(VAL_DATASET)
    val_lst_main = [t.rsplit('.', 1)[0] for t in val_lst_main_ext]
    test_lst_main_ext = get_main_dataset_files(TEST_DATASET)
    test_lst_main = [t.rsplit('.', 1)[0] for t in test_lst_main_ext]
    
    # Divide the samples files into training set, validation and test sets
    train_lst, val_lst, test_lst = split_dataset_files(DATASET)
#    print("SEQ_SIZE : {}".format(SEQ_SIZE))
    
#    # Write targets for list of Highlight videos 
#    for video_prefix in train_lst:
#        write_target(video_prefix, DATASET, LABELS)
#    
#    for video_prefix in train_lst_main:
#        write_target(video_prefix, MAIN_DATASET, MAIN_LABELS)
#
#    for video_prefix in val_lst_main:
#        write_target(video_prefix, VAL_DATASET, VAL_LABELS)
#        
#    for video_prefix in test_lst_main:
#        write_target(video_prefix, TEST_DATASET, TEST_LABELS)

    train_lst_main.extend(train_lst)
    # generate JSON file with config information
    class_index = ["non-stroke", "stroke"]
    write_data_info(class_index, train_lst_main, test_lst_main)    
    