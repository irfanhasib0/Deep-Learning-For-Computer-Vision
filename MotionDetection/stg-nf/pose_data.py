import os
import json
import cv2
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

class GeneratePoseData():
    def __init__(self,person_json_root):
        self.start_ofst  = 0
        self.seg_stride  = 2
        self.seg_len     = 24
        self.headless    = False
        self.seg_conf_th = 0.0
        self.dataset          = "ShanghaiTech"
        self.person_json_root = person_json_root
        SHANGHAITECH_HR_SKIP  = [(1, 130), (1, 135), (1, 136), (6, 144), (6, 145), (12, 152)]
    
    def shanghaitech_hr_skip(self,shanghaitech_hr, scene_id, clip_id):
        if not shanghaitech_hr:
            return shanghaitech_hr
        if (int(scene_id), int(clip_id)) in SHANGHAITECH_HR_SKIP:
            return True
        return False

    def gen_dataset(self,num_clips=None, kp18_format=True, ret_keys=False, ret_global_data=True):
        segs_data_np  = []
        segs_score_np = []
        segs_meta     = []

        start_ofst  = self.start_ofst
        seg_stride  = self.seg_stride
        seg_len     = self.seg_len
        headless    = self.headless
        seg_conf_th = self.seg_conf_th
        
        dir_list = os.listdir(self.person_json_root)
        json_list = sorted([fn for fn in dir_list if fn.endswith('tracked_person.json')])
        
        if num_clips is not None:
            json_list = [json_list[num_clips]]  # For debugging purposes
        
        for person_dict_fn in tqdm(json_list[:5]):
            if self.dataset == "UBnormal":
                type, scene_id, clip_id = \
                    re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*)_alphapose_.*', person_dict_fn)[0]
                clip_id = type + "_" + clip_id
            else:
                scene_id, clip_id = person_dict_fn.split('_')[:2]
                if self.shanghaitech_hr_skip(self.dataset=="ShaghaiTech-HR", scene_id, clip_id):
                    continue
            clip_json_path = os.path.join(self.person_json_root, person_dict_fn)
            with open(clip_json_path, 'r') as f:
                clip_dict = json.load(f)
            clip_segs_data_np, clip_segs_meta,score_segs_data_np = self.gen_clip_seg_data_np(
                clip_dict, 
                start_ofst,
                seg_stride,
                seg_len,
                scene_id=scene_id,
                clip_id=clip_id,
                ret_keys=ret_keys)

            segs_data_np.append(clip_segs_data_np)
            segs_score_np.append(score_segs_data_np)
            segs_meta += clip_segs_meta

        # Global data
        segs_data_np   = np.concatenate(segs_data_np, axis=0)
        segs_score_np  = np.concatenate(segs_score_np, axis=0)

        # if normalize_pose_segs:
        #     segs_data_np = normalize_pose(segs_data_np, vid_res=vid_res, **dataset_args)
        #     global_data_np = normalize_pose(global_data_np, vid_res=vid_res, **dataset_args)
        #     global_data = [normalize_pose(np.expand_dims(data, axis=0), **dataset_args).squeeze() for data
        #                    in global_data]
        if kp18_format and segs_data_np.shape[-2] == 17:
            segs_data_np   = self.keypoints17_to_coco18(segs_data_np)
        if headless:
            segs_data_np   = segs_data_np[:, :, 5:]

        #segs_data_np   = np.transpose(segs_data_np, (0, 3, 1, 2)).astype(np.float32)

        if seg_conf_th > 0.0:
            segs_data_np, segs_meta, segs_score_np = \
                self.seg_conf_th_filter(segs_data_np, segs_meta, segs_score_np, seg_conf_th)

        return segs_data_np, segs_meta, segs_score_np


    def keypoints17_to_coco18(self,kps):
        """
        Convert a 17 keypoints coco format skeleton to an 18 keypoint one.
        New keypoint (neck) is the average of the shoulders, and points
        are also reordered.
        """
        kp_np = np.array(kps)
        neck_kp_vec = 0.5 * (kp_np[..., 5, :] + kp_np[..., 6, :])
        kp_np = np.concatenate([kp_np, neck_kp_vec[..., None, :]], axis=-2)
        opp_order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        opp_order = np.array(opp_order, dtype=np.int32)
        kp_coco18 = kp_np[..., opp_order, :]
        return kp_coco18

    def seg_conf_th_filter(self,segs_data_np, segs_meta, segs_score_np, seg_conf_th=2.0):
        # seg_len = segs_data_np.shape[2]
        # conf_vals = segs_data_np[:, 2]
        # sum_confs = conf_vals.sum(axis=(1, 2)) / seg_len
        sum_confs = segs_score_np.mean(axis=1)
        seg_data_filt = segs_data_np[sum_confs > seg_conf_th]
        seg_meta_filt = list(np.array(segs_meta)[sum_confs > seg_conf_th])
        segs_score_np = segs_score_np[sum_confs > seg_conf_th]

        return seg_data_filt, seg_meta_filt, segs_score_np


    def gen_clip_seg_data_np(self,clip_dict, start_ofst=0, seg_stride=4, seg_len=12, scene_id='', clip_id='', ret_keys=False,global_pose_data=[]):
        """
        Generate an array of segmented sequences, each object is a segment and a corresponding metadata array
        """
        pose_segs_data  = []
        score_segs_data = []
        pose_segs_meta  = []
        
        for idx in sorted(clip_dict.keys(), key=lambda x: int(x)):
            sing_pose_np, sing_pose_meta, sing_pose_keys, sing_scores_np = self.single_pose_dict2np(clip_dict, idx)
            curr_pose_segs_np, curr_pose_segs_meta, curr_pose_score_np   = self.split_pose_to_segments(sing_pose_np,
                                                                                                sing_pose_meta,
                                                                                                sing_pose_keys,
                                                                                                start_ofst, seg_stride,
                                                                                                seg_len,
                                                                                                scene_id=scene_id,
                                                                                                clip_id=clip_id,
                                                                                                single_score_np=sing_scores_np)
            
            pose_segs_data.append(curr_pose_segs_np)
            score_segs_data.append(curr_pose_score_np)
            pose_segs_meta+= curr_pose_segs_meta
            
        if len(pose_segs_data) == 0:
            pose_segs_data_np = np.empty(0).reshape(0, seg_len, 17, 3)
            score_segs_data_np = np.empty(0).reshape(0, seg_len)
        else:
            pose_segs_data = np.concatenate(pose_segs_data, axis=0)
            score_segs_data = np.concatenate(score_segs_data, axis=0)
       
        return pose_segs_data, pose_segs_meta, score_segs_data


    def single_pose_dict2np(self,person_dict, idx):
        single_person = person_dict[str(idx)]
        sing_pose_np = []
        sing_scores_np = []
        if isinstance(single_person, list):
            single_person_dict = {}
            for sub_dict in single_person:
                single_person_dict.update(**sub_dict)
            single_person = single_person_dict
        single_person_dict_keys = sorted(single_person.keys())
        sing_pose_meta = [int(idx), int(single_person_dict_keys[0])]  # Meta is [index, first_frame]
        for key in single_person_dict_keys:
            curr_pose_np = np.array(single_person[key]['keypoints']).reshape(-1, 3)
            sing_pose_np.append(curr_pose_np)
            sing_scores_np.append(single_person[key]['scores'])
        sing_pose_np = np.stack(sing_pose_np, axis=0)
        sing_scores_np = np.stack(sing_scores_np, axis=0)
        return sing_pose_np, sing_pose_meta, single_person_dict_keys, sing_scores_np

    def split_pose_to_segments(self,single_pose_np, single_pose_meta, single_pose_keys, start_ofst=0, seg_dist=6, seg_len=12,
                           scene_id='', clip_id='', single_score_np=None, dataset="ShanghaiTech"):
        clip_t, kp_count, kp_dim = single_pose_np.shape
        pose_segs_np   = np.empty([0, seg_len, kp_count, kp_dim])
        pose_score_np  = np.empty([0, seg_len])
        pose_segs_meta = []
        num_segs = np.ceil((clip_t - seg_len) / seg_dist).astype(np.int32)
        single_pose_keys_sorted = sorted([int(i) for i in single_pose_keys])  # , key=lambda x: int(x))
        
        for seg_ind in range(num_segs):
            start_ind = start_ofst + seg_ind * seg_dist
            start_key = single_pose_keys_sorted[start_ind]
            
            if self.is_seg_continuous(single_pose_keys_sorted, start_key, seg_len):
                curr_segment    = single_pose_np[start_ind:start_ind + seg_len].reshape(1, seg_len, kp_count, kp_dim)
                curr_score      = single_score_np[start_ind:start_ind + seg_len].reshape(1, seg_len)
                pose_segs_np    = np.append(pose_segs_np, curr_segment, axis=0)
                pose_score_np   = np.append(pose_score_np, curr_score, axis=0)
                pose_segs_meta.append([int(scene_id), int(clip_id), int(single_pose_meta[0]), int(start_key)])
                
        return pose_segs_np, pose_segs_meta, pose_score_np
    
    def is_seg_continuous(self,sorted_seg_keys, start_key, seg_len, missing_th=2):
        """
        Checks if an input clip is continuous or if there are frames missing
        :param sorted_seg_keys:
        :param start_key:
        :param seg_len:
        :param missing_th: The number of frames that are allowed to be missing on a sequence,
        i.e. if missing_th = 1 then a seg for which a single frame is missing is considered continuous
        :return:
        """
        start_idx = sorted_seg_keys.index(start_key)
        expected_idxs = list(range(start_key, start_key + seg_len))
        act_idxs = sorted_seg_keys[start_idx: start_idx + seg_len]
        min_overlap = seg_len - missing_th
        key_overlap = len(set(act_idxs).intersection(expected_idxs))
        if key_overlap >= min_overlap:
            return True
        else:
            return False

        
class PoseGraph(Dataset):
    def __init__(self):
        self.edge_index = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
              [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17]]
        
        json_dir  = '/home/irfan/Desktop/Data/shanghaitech/pose/test/' 
        frame_dir = '/home/irfan/Desktop/Data/shanghaitech/testing/frames/'
        
        dataset = GeneratePoseData(json_dir)
        self.data, self.metadata, self.score = dataset.gen_dataset()
    
    def __getitem__(self,index):
        return self.data[index].copy(),self.metadata[index].copy()
    
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def get_norm(dt):
        dt[:,:,0]-=dt[:,:,0].min()
        dt[:,:,1]-=dt[:,:,1].min()
        dt[:,:,0]/=dt[:,:,0].max()
        dt[:,:,1]/=dt[:,:,1].max()
        dt[:,:,2]-=dt[:,:,2].min()
        dt[:,:,2]/=dt[:,:,2].max()
        return dt
    
    @staticmethod
    def get_polar(dt):
        x = np.sqrt(dt[:,:,0]**2 + dt[:,:,1]**2)
        y = np.arctan2(dt[:,:,0],dt[:,:,1])
        return x,y
    
    def plot_pose(self,idx,_id=list(range(18))):
        _dt,mtd = self.__getitem__(idx)
        mtd=list(map(str,mtd))
        fname = f"{mtd[0].zfill(2)}_{mtd[1].zfill(4)}/{mtd[3].zfill(3)}.jpg"
        src = '/home/irfan/Desktop/Data/shanghaitech/testing/frames/'+fname
        img = cv2.imread(src)
        fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(6,3))
        axes[0].imshow(img)
        for i in range(24):
            axes[1].scatter(_dt[i,_id,0],_dt[i,_id,1],marker='.')
        axes[1].imshow(img)
        plt.show()
        plt.close()
        
    def plot_hmap(self,idx):
        dt,mtd = self.__getitem__(idx)
        #_dt    = dt.transpose(1,2,0)
        dt     = self.get_norm(dt)
        x,y    = self.get_polar(dt)
        
        fig,axes = plt.subplots(nrows=1,ncols=5,figsize=(10,3))
        axes[0].imshow(dt[:,:,0])
        axes[1].imshow(dt[:,:,1])
        axes[2].imshow(dt[:,:,2])
        axes[3].imshow(x)
        axes[4].imshow(y)
        for ax in axes:
            ax.set_xticks(range(11),rotation=90)
        plt.show()
        plt.close()
        
    def vis_data(self,idx,_id = list(range(18))):
        dt,mdt = self.__getitem__(idx)
        __dt = dt.copy()
        #__dt = self.get_norm(dt)
        __dt[:,:,0]  = dt[:,:,0] - dt[:,:,0].min()
        __dt[:,:,1]  = dt[:,:,1] - dt[:,:,1].min()
        ofs  = int(__dt[:,:,0].max()//5)
        size = __dt.max().astype('int32')
        fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(10,3))
        for i in _id:
            axes[0].scatter(ofs+__dt[:,i,0],ofs+__dt[:,i,1],marker='.')
        axes[0].imshow(np.zeros((size+2*ofs,size+2*ofs,3),dtype=np.uint8)+255)
        ofs  = int(__dt[:,:,1].max())
        for i in range(24):
            axes[1].scatter(__dt[i,_id,0]+ofs*i, ofs+__dt[i,_id,1],marker='.')
        axes[1].imshow(np.zeros((size+2*ofs,size+ofs*i,3),dtype=np.uint8)+255)
        plt.show()
        plt.close()