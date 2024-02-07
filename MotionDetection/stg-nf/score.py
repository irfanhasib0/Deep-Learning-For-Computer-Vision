import os
import re
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from tqdm import tqdm



    
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Score():
    def __init__(self,dataset='ShanghaiTech-HR',path='/home/irfan/Desktop/Data/Pose_JSON_Data/ShanghaiTech/gt/test_frame_mask/'):
        self.dataset = dataset
        self.seg_stride  = 2
        self.seg_len     = 24
        self.SHANGHAITECH_HR_SKIP = [(1, 130), (1, 135), (1, 136), (6, 144), (6, 145), (12, 152)]
        self.per_frame_scores_root = path
        
    def score_dataset(self,score, metadata):
        gt_arr, scores_arr = self.get_dataset_scores(score, metadata)
        scores_arr         = self.smooth_scores(scores_arr)
        gt_np              = np.concatenate(gt_arr)
        scores_np          = np.concatenate(scores_arr)
        auc                = self.score_auc(scores_np, gt_np)
        return auc, scores_np


    def get_dataset_scores(self,scores, metadata):
        dataset_gt_arr = []
        dataset_scores_arr = []

        if self.dataset == 'UBnormal':
            pose_segs_root = 'data/UBnormal/pose/test'
            clip_list = os.listdir(pose_segs_root)
            clip_list = sorted(
                fn.replace("alphapose_tracked_person.json", "tracks.txt") for fn in clip_list if fn.endswith('.json'))
            per_frame_scores_root = 'data/UBnormal/gt/'
        else:
            clip_list = os.listdir(self.per_frame_scores_root)
            clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))

        print("Scoring {} clips".format(len(clip_list)))
        for clip in tqdm(clip_list):
            clip_gt, clip_score = self.get_clip_score(scores, metadata, clip)
            if clip_score is not None:
                dataset_gt_arr.append(clip_gt)
                dataset_scores_arr.append(clip_score)

        scores_np = np.concatenate(dataset_scores_arr, axis=0)
        scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max()
        scores_np[scores_np == -1 * np.inf] = scores_np[scores_np != -1 * np.inf].min()
        index = 0
        for score in range(len(dataset_scores_arr)):
            for t in range(dataset_scores_arr[score].shape[0]):
                dataset_scores_arr[score][t] = scores_np[index]
                index += 1

        return dataset_gt_arr, dataset_scores_arr


    def score_auc(self,scores_np, gt):
        scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max()
        scores_np[scores_np == -1 * np.inf] = scores_np[scores_np != -1 * np.inf].min()
        scores_np = np.nan_to_num(scores_np)
        auc       = roc_auc_score(gt, scores_np)
        return auc


    def smooth_scores(self,scores_arr, sigma=7):
        for s in range(len(scores_arr)):
            for sig in range(1, sigma):
                scores_arr[s] = gaussian_filter1d(scores_arr[s], sigma=sig)
        return scores_arr

    ##dists      = ((kmn.cluster_centers_[kmn.predict(mu)] - mu)**2).mean(axis=1)
    def get_clip_score(self,all_scores, all_metadata, clip, score_func = lambda x:x):
        if self.dataset == 'UBnormal':
            type, scene_id, clip_id = re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*)_tracks.*', clip)[0]
            clip_id = type + "_" + clip_id
        else:
            scene_id, clip_id = [int(i) for i in clip.replace("label", "001").split('.')[0].split('_')]
            if self.shanghaitech_hr_skip((self.dataset == 'ShanghaiTech-HR'), scene_id, clip_id):
                return None, None
        
        mtd_inds = np.where((all_metadata[:, 1] == clip_id) &
                            (all_metadata[:, 0] == scene_id))[0]
        _metadata = all_metadata[mtd_inds]
        _scores   = all_scores[mtd_inds]
        
        fig_idxs = set(_metadata[:,2])
        res_fn   = os.path.join(self.per_frame_scores_root, clip)
        gt       = np.load(res_fn)
    
        #if self.dataset != "UBnormal":
        #    gt = np.ones(gt.shape) - gt  # 1 is normal, 0 is abnormal
        
        min_scores = -np.inf * np.ones(gt.shape[0])
        if len(fig_idxs) == 0:
            scores_dict = {0: np.copy(min_scores)}
        else:
            scores_dict = {i: np.copy(min_scores) for i in fig_idxs}
        #import pdb; pdb.set_trace()
        for person_id in fig_idxs:
            pid_inds = np.where((_metadata[:, 1] == clip_id) &
                                (_metadata[:, 0] == scene_id) &
                                (_metadata[:, 2] == person_id))[0]
            pid_scores     = _scores[pid_inds]
            #pid_scores = score_func(pid_scores)
            pid_frame_inds = _metadata[pid_inds,3].astype(int)
            
            scores_dict[person_id][pid_frame_inds + int((self.seg_stride*self.seg_len) / 2)] = pid_scores
        
        score_arr = np.stack(list(scores_dict.values()))
        
        score = np.amax(score_arr, axis=0) #amin
        if len(fig_idxs) : 
            score[score == -np.inf] = min(0,score[score != -np.inf].min())
        return gt, score
    
    def shanghaitech_hr_skip(self,shanghaitech_hr, scene_id, clip_id):
        if not shanghaitech_hr:
            return shanghaitech_hr
        if (int(scene_id), int(clip_id)) in self.SHANGHAITECH_HR_SKIP:
            return True
        return False
