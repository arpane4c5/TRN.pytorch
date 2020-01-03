import os
import os.path as osp
import sys
import time
import json

import torch
import torch.nn as nn
import numpy as np

import _init_paths
import utils as utl
from configs.cricket import parse_trn_args as parse_args
from models import build_model

# Local Paths
VAL_DATASET = "/home/arpan/VisionWorkspace/Cricket/dataset_25_fps_test_set"
VAL_LABELS = "/home/arpan/VisionWorkspace/Cricket/dataset_25_fps_test_set_labels"
# Server Paths
if os.path.exists("/opt/datasets/cricket/ICC_WT20"):
    VAL_DATASET = "/home/arpan/DATA_Drive/Cricket/dataset_25_fps_test_set"
    VAL_LABELS = "/home/arpan/DATA_Drive/Cricket/dataset_25_fps_test_set_labels"

def to_device(x, device):
    return x.unsqueeze(0).to(device)

def getScoredLocalizations(enc_score_metrics, class_index=1, seq_diff=0, threshold=0.9):
    """
    Receives a list of class scores and returns the localizations for the 
    action corresponding to the class index.
    
    Parameters:
    ----------
    enc_score_metrics : list of (1, nClasses) arrays of probabilities.
    class_index : int 
    
    """
    
    enc_score_metrics = np.array(enc_score_metrics)[:, class_index]

    # generate binary predictions
    predictions = np.zeros_like(enc_score_metrics)
    predictions[enc_score_metrics >= threshold] = 1
    
    segments = getVidLocalizations(predictions)
    scores = [sum(enc_score_metrics[beg:end]) for (beg, end) in segments]
    
    # shift the segments
    segments = [(beg+seq_diff, end+seq_diff) for (beg, end) in segments]
    return {"segments": segments, "scores": scores}
    
    
def getVidLocalizations(binaryPreds): 
    """
    Receive a list of binary predictions and generate the action localizations
    for a video
    """
    vLocalizations = []
    act_beg, act_end = -1, -1
    isAction = False
    for i,pred in enumerate(binaryPreds):
        if not isAction:
            if pred == 1:  # Transition from non-action to action
                isAction = True
                act_beg = i
            # if p==0: # skip since it is already non-action
        else:           # if action is going on
            if pred == 0:    # Transition from action to non-action
                isAction = False
                act_end = (i-1)
                # Append to actions list
                vLocalizations.append((act_beg, act_end))
                act_beg, act_end = -1, -1   # Reset
                
    if isAction and act_beg != -1:
        act_end = len(binaryPreds) -1
        vLocalizations.append((act_beg, act_end))
        
    return vLocalizations


# function to remove the action segments that have less than "epsilon" frames.
def filter_action_segments(shots_dict, epsilon=10):
    filtered_shots = {}
    for k,v in shots_dict.items():
        vsegs = []
        vscores = []
        for idx, segment in enumerate(v["segments"]):
            if (segment[1]-segment[0] >= epsilon):
                vsegs.append(segment)
                vscores.append(v["scores"][idx])
        filtered_shots[k] = {"segments":vsegs, "scores":vscores}
    return filtered_shots


def main(args):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    enc_score_metrics = []
    enc_target_metrics = []
    dec_score_metrics = [[] for i in range(args.dec_steps)]
    dec_target_metrics = [[] for i in range(args.dec_steps)]

    if osp.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
    else:
        raise(RuntimeError('Cannot find the checkpoint {}'.format(args.checkpoint)))
    model = build_model(args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train(False)

    softmax = nn.Softmax(dim=1).to(device)
    localizations = {}

    for session_idx, session in enumerate(args.test_session_set, start=1):
        start = time.time()
        with torch.set_grad_enabled(False):
            print("ID {}: {}".format(session_idx, session))
            camera_inputs = np.load(osp.join(args.data_root, args.camera_feature, session+'.npy'))
            motion_inputs = np.load(osp.join(args.data_root, args.motion_feature, session+'.npy'))
            # For Cricket
            camera_inputs = np.squeeze(camera_inputs, axis=1)
            motion_inputs = np.squeeze(motion_inputs, axis=1)
            camera_inputs[camera_inputs==float("-Inf")] = 0
            camera_inputs[camera_inputs==float("Inf")] = 0
            motion_inputs[motion_inputs==float("-Inf")] = 0
            motion_inputs[motion_inputs==float("Inf")] = 0
            
            target = np.load(osp.join(args.data_root, 'target', session+'.npy'))
            future_input = to_device(torch.zeros(model.future_size), device)
            enc_hx = to_device(torch.zeros(model.hidden_size), device)
            enc_cx = to_device(torch.zeros(model.hidden_size), device)
            
            niters = min([motion_inputs.shape[0], camera_inputs.shape[0]])
            seq_diff = camera_inputs.shape[0] - motion_inputs.shape[0]
            vid_enc_score_metrics = []

            for l in range(niters):
                camera_input = to_device(
                    torch.as_tensor(camera_inputs[l+seq_diff].astype(np.float32)), device)
                motion_input = to_device(
                    torch.as_tensor(motion_inputs[l].astype(np.float32)), device)

                future_input, enc_hx, enc_cx, enc_score, dec_score_stack = \
                        model.step(camera_input, motion_input, future_input, enc_hx, enc_cx)

                vid_enc_score_metrics.append(softmax(enc_score).cpu().numpy()[0])
                enc_target_metrics.append(target[l+seq_diff, 1])

                for step in range(args.dec_steps):
                    dec_score_metrics[step].append(softmax(dec_score_stack[step]).cpu().numpy()[0])
                    dec_target_metrics[step].append(target[min(l + step, target.shape[0] - 1), 1])
        end = time.time()
        
        enc_score_metrics.extend(vid_enc_score_metrics)

        print('Processed session {}, {:2} of {}, running time {:.2f} sec'.format(
            session, session_idx, len(args.test_session_set), end - start))
        
        if osp.isfile(osp.join(VAL_DATASET, session+'.mp4')):
            session = session + '.mp4'
        else:
            session = session + '.avi'
            
        if session.startswith("v_"):
            localizations["youtube/"+session] = getScoredLocalizations(vid_enc_score_metrics, 1)
        elif session.startswith("IPL2017"):
            localizations["ipl2017/"+session] = getScoredLocalizations(vid_enc_score_metrics, 1)
        elif session.startswith("Game "):
            localizations["cpl2015/"+session] = getScoredLocalizations(vid_enc_score_metrics, 1)
        else:
            localizations["hotstar/"+session] = getScoredLocalizations(vid_enc_score_metrics, 1)
        

    save_dir = osp.dirname(args.checkpoint)
    result_file  = osp.basename(args.checkpoint).replace('.pth', '.json')
    # Compute result for encoder
    utl.compute_result(args.class_index, enc_score_metrics, enc_target_metrics, \
                save_dir, result_file, ignore_class=[0], save=True, verbose=True)

    # Compute result for decoder
    for step in range(args.dec_steps):
        utl.compute_result(args.class_index, dec_score_metrics[step], \
                           dec_target_metrics[step], save_dir, result_file, \
                           ignore_class=[0], save=False, verbose=True)
    
    #print(localizations)
    with open("prediction_localizations_TRN_mainVal.json", "w") as fp:
        json.dump(localizations, fp)

if __name__ == '__main__':
    #main(parse_args())
    
    with open("prediction_localizations_TRN_hlVal_C3D17_hidden1024_Th09_ep50.json", "r") as fp:
        localizations = json.load(fp)
        
    i = 60  # optimum
    filtered_shots = filter_action_segments(localizations, epsilon=i)
        
    #print(localizations)
    with open("prediction_localizations_TRN_mainVal.json", "w") as fp:
        json.dump(filtered_shots, fp)
