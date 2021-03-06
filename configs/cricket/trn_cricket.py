from configs import parse_base_args, build_data_info

__all__ = ['parse_trn_args']

def parse_trn_args():
    parser = parse_base_args()
    parser.add_argument('--data_root', default='data/CRICKET', type=str)
    parser.add_argument('--model', default='TRN', type=str)
    parser.add_argument('--inputs', default='motion', type=str)
    parser.add_argument('--hidden_size', default=1024, type=int)
#    parser.add_argument('--camera_feature', default='resnet200-fc', type=str)
#    parser.add_argument('--motion_feature', default='bn-inception', type=str)
#    parser.add_argument('--camera_feature', default='hog_64x64', type=str)
    parser.add_argument('--camera_feature', default='farneback_flow_grid20_main', type=str)
    #parser.add_argument('--motion_feature', default='farneback_flow_grid20', type=str)    
#    parser.add_argument('--motion_feature', default='c3dFinetuned_feats_17', type=str)    
    parser.add_argument('--motion_feature', \
                default='c3dFinetunedOnHLMainSeq23_mainDataset_feats_17', type=str)
    parser.add_argument('--enc_steps', default=64, type=int)
    parser.add_argument('--dec_steps', default=8, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    return build_data_info(parser.parse_args())
