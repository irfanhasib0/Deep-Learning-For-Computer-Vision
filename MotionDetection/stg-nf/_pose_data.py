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
        self.dataset          = "ShanghaiTech-HR"
        self.person_json_root = person_json_root
        SHANGHAITECH_HR_SKIP  = [(1, 130), (1, 135), (1, 136), (6, 144), (6, 145), (12, 152)]
        
        self.dtype_pose       = np.uint16
        self.dtype_pose_score = np.float16
        self.dtype_prop_score = np.float16
        self.dtype_meta_data  = np.uint16
        
        self.poses       = np.empty((0, self.seg_len, 17, 2), dtype=self.dtype_pose)
        self.pose_scores = np.empty((0, self.seg_len, 17),    dtype=self.dtype_pose_score)
        self.prop_scores = np.empty((0, self.seg_len),        dtype=self.dtype_prop_score)
        self.meta_data   = np.empty((0,4),                    dtype=self.dtype_meta_data)
        
    def shanghaitech_hr_skip(self,shanghaitech_hr, scene_id, clip_id):
        if not shanghaitech_hr:
            return shanghaitech_hr
        if (int(scene_id), int(clip_id)) in SHANGHAITECH_HR_SKIP:
            return True
        return False
    
    def get_scene_id(self,json_file):
        if self.dataset == "UBnormal":
                re_expr = '(abnormal|normal)_scene_(\d+)_scenario(.*)_alphapose_.*'
                _type, scene_id, clip_id = re.findall(re_expr, json_file)[0]
                clip_id = _type + "_" + clip_id
        else:
            scene_id, clip_id = json_file.split('_')[:2]
            if self.shanghaitech_hr_skip(self.dataset=="ShaghaiTech-HR", scene_id, clip_id):
                scene_id = -1
                clip_id  = -1
        return scene_id, clip_id
                    
    def gen_dataset(self, ret_keys=False, ret_global_data=True):
        poses       = []
        pose_scores = []
        prop_scores = []
        meta_data   = []

        start_ofst  = self.start_ofst
        seg_stride  = self.seg_stride
        seg_len     = self.seg_len
        seg_conf_th = self.seg_conf_th
        
        dir_list  = os.listdir(self.person_json_root)
        json_list = sorted([fn for fn in dir_list if fn.endswith('tracked_person.json')])
        
        for json_file in tqdm(json_list):
            scene_id , clip_id = self.get_scene_id(json_file)
            if scene_id == -1: continue
            
            clip_json_path = os.path.join(self.person_json_root, json_file)
            with open(clip_json_path, 'r') as f:
                clip_dict = json.load(f)
            poses, pose_scores, prop_scores, meta_data = self.gen_clip_seg_data(clip_dict,
                                   scene_id = scene_id,
                                   clip_id  = clip_id,
                                   ret_keys = ret_keys)

            #poses.append(clip_poses)
            #pose_scores.append(clip_pose_scores)
            #prop_scores.append(clip_prop_scores)
            #meta_data.append(clip_meta_data)
            
            self.poses       = np.append(self.poses,poses)
            self.pose_scores = np.append(self.pose_scores,pose_scores)
            self.prop_scores = np.append(self.prop_scores,prop_scores)
            self.meta_data   = np.append(self.meta_data,meta_data)
        
        #poses       = np.concatenate(poses,       axis=0, dtype=self.dtype_pose)
        #pose_scores = np.concatenate(pose_scores, axis=0, dtype=self.dtype_pose_score)
        #prop_scores = np.concatenate(prop_scores, axis=0, dtype=self.dtype_prop_score)
        #meta_data   = np.concatenate(meta_data,   axis=0, dtype=self.dtype_meta_data)

        # if normalize_pose_segs:
        #     segs_data_np = normalize_pose(segs_data_np, vid_res=vid_res, **dataset_args)
        
        if self.poses.shape[-2] == 17: 
            self.poses       = self.keypoints17_to_coco18(self.poses)
            self.pose_scores = self.keypoints17_to_coco18(self.pose_scores[:,:,:,None])
        #segs_data_np   = np.transpose(segs_data_np, (0, 3, 1, 2)).astype(np.float32)

        #if seg_conf_th > 0.0:
        #    segs_data_np, segs_meta, segs_score_np = \
        #        self.seg_conf_th_filter(segs_data_np, segs_meta, segs_score_np, seg_conf_th)

        return self.poses, self.pose_scores, self.prop_scores, self.meta_data


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


    def gen_clip_seg_data(self,
                          clip_dict,
                          scene_id='', 
                          clip_id='', 
                          ret_keys=False,
                          global_pose_data=[]):
        
        poses       = []
        pose_scores = []
        prop_scores = []
        meta_data   = []
        
        for idx in sorted(clip_dict.keys(), key=lambda x: int(x)):
            sp_poses, sp_pose_scores, sp_prop_scores, sp_frame_ids, sp_meta = self.single_person_poses(clip_dict, idx)
            _poses, _pose_scores, _prop_scores, _meta_data = self.split_pose_to_segments(sp_poses,
                                        sp_pose_scores,
                                        sp_prop_scores,
                                        sp_meta,
                                        sp_frame_ids,
                                        scene_id  = scene_id,
                                        clip_id   = clip_id)
            
            poses.append(_poses)
            pose_scores.append(_pose_scores)
            prop_scores.append(_prop_scores)
            meta_data.append(_meta_data)
        
        poses       = np.concatenate(poses,       axis=0)
        pose_scores = np.concatenate(pose_scores, axis=0)
        prop_scores = np.concatenate(prop_scores, axis=0)
        meta_data   = np.concatenate(meta_data,   axis=0)
        
        return poses, pose_scores, prop_scores, meta_data


    def single_person_poses(self,clip_dict, idx):
        
        sp_dict        = clip_dict[str(idx)]
        sp_frame_ids   = sorted(sp_dict.keys())
        
        sp_poses       = np.zeros((len(sp_frame_ids),17,2), dtype=self.dtype_pose)
        sp_pose_scores = np.zeros((len(sp_frame_ids),17),   dtype=self.dtype_pose_score)
        sp_prop_scores = np.zeros((len(sp_frame_ids)),      dtype=self.dtype_prop_score)
        sp_meta        = [int(idx), int(sp_frame_ids[0])]  # Meta is [index, first_frame]
        
        for i,frame_id in enumerate(sp_frame_ids):
            curr_pose_np = np.array(sp_dict[frame_id]['keypoints']).reshape(-1, 3)
            sp_poses[i]       = curr_pose_np[:,:2]
            sp_pose_scores[i] = curr_pose_np[:,2]
            sp_prop_scores[i] = sp_dict[frame_id]['scores']
        
        return sp_poses, sp_pose_scores, sp_prop_scores, sp_frame_ids, sp_meta

    def split_pose_to_segments(self,
                               sp_poses,
                               sp_pose_scores,
                               sp_prop_scores,
                               sp_meta, 
                               single_frame_ids,
                               scene_id   = '', 
                               clip_id    = ''):
        
        clip_t, kp_count, kp_dim = sp_poses.shape
        
        poses       = np.empty((0, self.seg_len, 17, 2), dtype=self.dtype_pose)
        pose_scores = np.empty((0, self.seg_len, 17),    dtype=self.dtype_pose_score)
        prop_scores = np.empty((0, self.seg_len),        dtype=self.dtype_prop_score)
        meta_data   = np.empty((0,4),                    dtype=self.dtype_meta_data)
        
        num_segs    = np.ceil((clip_t - self.seg_len) / self.seg_stride).astype(np.int32)    
        single_frame_ids_sorted = sorted([int(i) for i in single_frame_ids])  # , key=lambda x: int(x))
        
        for seg_ind in range(num_segs):
            start_ind = self.start_ofst + seg_ind * self.seg_stride
            start_key = single_frame_ids_sorted[start_ind]
            
            if self.is_seg_continuous(single_frame_ids_sorted, start_key, self.seg_stride):
                meta_data   = np.array([[int(scene_id), int(clip_id), int(sp_meta[0]), int(start_key)]],dtype=self.dtype_meta_data)
                poses       = np.append(poses,sp_poses[start_ind:start_ind + self.seg_len][None],axis=0)
                pose_scores = np.append(pose_scores,sp_pose_scores[start_ind:start_ind + self.seg_len][None],axis=0)
                prop_scores = np.append(prop_scores,sp_prop_scores[start_ind:start_ind + self.seg_len][None],axis=0)
                meta_data   = np.append(meta_data,meta_data,axis=0)
                
        return poses, pose_scores, prop_scores, meta_data
    
    def is_seg_continuous(self,sorted_seg_keys, start_key, seg_len, missing_th=2):
        
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
    def __init__(self,json_dir):
        self.edge_index = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
              [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17]]
        
        #json_dir  = '/home/irfan/Desktop/Data/shanghaitech/pose/test/' 
        frame_dir = '/home/irfan/Desktop/Data/shanghaitech/testing/frames/'
        
        dataset = GeneratePoseData(json_dir)
        self.poses, self.pose_scores, self.prop_scores, self.meta_data = dataset.gen_dataset()
    
    def __getitem__(self,index):
        data = np.concatenate([self.poses[index], self.pose_scores[index]],axis=-1).copy()
        return data.astype('float32') , self.meta_data[index].copy()
    
    def __len__(self):
        return len(self.poses)
    
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