import os
import json
import glob
import cv2
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

def normalize_pose(pose_data, args):
   
    norm_factor    = np.array(args['vid_res'])
    pose_data = pose_data / norm_factor
    
    #pose_data[..., :2] = 2 * pose_data[..., :2] - 1
    pose_data[..., :2] = (pose_data[..., :2] - pose_data[..., :2].mean(axis=(0, 1))[None, None, :]) / pose_data[..., 1].std(axis=(0, 1))[None, None, None]
    
    return pose_data


class GeneratePoseData():
    def __init__(self,args,split='valid'):
        self.no_of_files = args['no_of_files']
        self.start_ofst  = 0
        self.seg_stride  = args['seg_stride']
        self.seg_len     = args['seg_len']
        self.headless    = False
        self.seg_conf_thr= args['seg_conf_thr']
        self.dataset          = "ShanghaiTech-HR"
        self.person_json_root = args['json_dir'][split]
        self.SHANGHAITECH_HR_SKIP  = [(1, 130), (1, 135), (1, 136), (6, 144), (6, 145), (12, 152)]
        
        self.dtype_pose       = np.float16
        self.dtype_pose_score = np.float16
        self.dtype_prop_score = np.float16
        self.dtype_meta_data  = np.uint16
        
        self.poses       = []#np.empty((0, self.seg_len, 17, 2), dtype=self.dtype_pose)
        self.pose_scores = []#np.empty((0, self.seg_len, 17),    dtype=self.dtype_pose_score)
        self.prop_scores = []#np.empty((0, self.seg_len),        dtype=self.dtype_prop_score)
        self.meta_data   = []#np.empty((0,4),                    dtype=self.dtype_meta_data)
        
        
    def shanghaitech_hr_skip(self,shanghaitech_hr, scene_id, clip_id):
        if not shanghaitech_hr:
            return shanghaitech_hr
        if (int(scene_id), int(clip_id)) in self.SHANGHAITECH_HR_SKIP:
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
        
        dir_list  = os.listdir(self.person_json_root)
        json_list = sorted([fn for fn in dir_list if fn.endswith('tracked_person.json')])
        
        for json_file in tqdm(json_list[:self.no_of_files]):
            scene_id , clip_id = self.get_scene_id(json_file)
            if scene_id == -1: continue
            
            clip_json_path = os.path.join(self.person_json_root, json_file)
            with open(clip_json_path, 'r') as f:
                clip_dict = json.load(f)
            self.gen_clip_seg_data(clip_dict,
                                   scene_id = scene_id,
                                   clip_id  = clip_id,
                                   ret_keys = ret_keys)

        # if normalize_pose_segs:
        #     segs_data_np = normalize_pose(segs_data_np, vid_res=vid_res, **dataset_args)
        
        self.poses       = np.concatenate(self.poses,axis=0)
        self.pose_scores = np.concatenate(self.pose_scores,axis=0)
        self.prop_scores = np.concatenate(self.prop_scores,axis=0)
        self.meta_data   = np.concatenate(self.meta_data,axis=0)
        
        if self.poses.shape[-2] == 17: 
            self.poses       = self.keypoints17_to_coco18(self.poses)
            self.pose_scores = self.keypoints17_to_coco18(self.pose_scores[:,:,:,None])
        #segs_data_np   = np.transpose(segs_data_np, (0, 3, 1, 2)).astype(np.float32)

        #if seg_conf_th > 0.0:
        #    segs_data_np, segs_meta, segs_score_np = \
        #        self.seg_conf_th_filter(segs_data_np, segs_meta, segs_score_np, self.seg_conf_th)

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
        
        for idx in sorted(clip_dict.keys(), key=lambda x: int(x)):
            sp_poses, sp_pose_scores, sp_prop_scores, sp_frame_ids, sp_meta = self.single_person_poses(clip_dict, idx)
            self.split_pose_to_segments(sp_poses,
                                        sp_pose_scores,
                                        sp_prop_scores,
                                        sp_meta,
                                        sp_frame_ids,
                                        scene_id  = scene_id,
                                        clip_id   = clip_id)
            

    def single_person_poses(self,clip_dict, idx):
        
        sp_dict        = clip_dict[str(idx)]
        sp_frame_ids   = sorted(sp_dict.keys())
        
        sp_poses       = np.zeros((len(sp_frame_ids),17,2), dtype=self.dtype_pose)
        sp_pose_scores = np.zeros((len(sp_frame_ids),17),   dtype=self.dtype_pose_score)
        sp_prop_scores = np.zeros((len(sp_frame_ids)),      dtype=self.dtype_prop_score)
        sp_meta        = [int(idx), int(sp_frame_ids[0])]  # Meta is [index, first_frame]
        
        for i,frame_id in enumerate(sp_frame_ids):
            curr_pose_np = np.array(sp_dict[frame_id]['keypoints']).reshape(-1, 3)
            sp_poses[i]       = np.clip(curr_pose_np[:,:2],0,4096)
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
        num_segs    = np.ceil((clip_t - self.seg_len) / self.seg_stride).astype(np.int32)    
        single_frame_ids_sorted = sorted([int(i) for i in single_frame_ids])  # , key=lambda x: int(x))
        
        for seg_ind in range(num_segs):
            start_ind = self.start_ofst + seg_ind * self.seg_stride
            start_key = single_frame_ids_sorted[start_ind]
            
            if self.is_seg_continuous(single_frame_ids_sorted, start_key, self.seg_len):
                meta_data        = np.array([int(scene_id), int(clip_id), int(sp_meta[0]), int(start_key)],dtype=self.dtype_meta_data)
                
                if sp_prop_scores[start_ind:start_ind + self.seg_len].mean() >= self.seg_conf_thr:
                    self.poses.append(sp_poses[start_ind:start_ind + self.seg_len][None])
                    self.pose_scores.append(sp_pose_scores[start_ind:start_ind + self.seg_len][None])
                    self.prop_scores.append(sp_prop_scores[start_ind:start_ind + self.seg_len][None])
                    self.meta_data.append(meta_data[None])
                
        
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
    def __init__(self,args,split='valid'):
        self.args       = args
        self.edge_index = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
              [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17]]
        
        self.channel   = args['channel']
        self.mask_dir  = args['gt_dir'][split]['mask_dir']
        self.frame_dir = args['gt_dir'][split]['frame_dir']
        
        dataset = GeneratePoseData(args,split=split)
        self.poses, self.pose_scores, self.prop_scores, self.meta_data = dataset.gen_dataset()
        # n,24,18,2
    def __getitem__(self,index,norm=True):
        data = np.concatenate([self.poses[index], self.pose_scores[index]],axis=-1).copy()
        data = data.astype('float32')[:,:,:self.channel]
        if norm: data = normalize_pose(data,self.args) 
        return data.astype('float32') , self.meta_data[index].astype('int16'), self.prop_scores[index].astype('float32')
    
    def __len__(self):
        return len(self.poses)
    
    def get_norm(self,dt):
        for i in range(dt.shape[2]):
            dt[:,:,i]-=dt[:,:,i].min()
            dt[:,:,i]/=dt[:,:,i].max()
            
        #for i in range(dt.shape[2]):
        #    dt[:,:,i]-=dt[:,:,i].min()
        #    dt[:,:,i]/=dt[:,:,i].std()
        #dt = np.clip(dt/5.0,-1.0,1.0)
        return dt
    
    def get_clip_poses(self,scene_id, clip_id):
        #if self.dataset == 'UBnormal':
        #    type, scene_id, clip_id = re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*)_tracks.*', clip)[0]
        #    clip_id = type + "_" + clip_id
        #else:
        #scene_id, clip_id = [int(i) for i in clip.replace("label", "001").split('.')[0].split('_')]
        #if self.shanghaitech_hr_skip((self.dataset == 'ShanghaiTech-HR'), scene_id, clip_id):
        #    return None, None
        clip_path = f"{str(scene_id).zfill(2)}_{str(clip_id).zfill(4)}"
        clip_metadata_inds = np.where((self.meta_data[:, 1] == clip_id) & (self.meta_data[:, 0] == scene_id))[0]
        clip_metadata      = self.meta_data[clip_metadata_inds]
        clip_fig_idxs      = set([arr[2] for arr in clip_metadata])
        try :
            clip_res_fn        = os.path.join(self.mask_dir, clip_path+'.npy')
            clip_gt            = np.load(clip_res_fn)
        except:
            clip_gt = 0
        #if self.dataset != "UBnormal":
        #    clip_gt = np.ones(clip_gt.shape) - clip_gt  # 1 is normal, 0 is abnormal
        
        clip_pose_dict = {}
        for person_id in clip_fig_idxs:
            _inds = np.where((self.meta_data[:, 1] == clip_id) & (self.meta_data[:, 0] == scene_id) & (self.meta_data[:, 2] == person_id))[0]
            frame_inds = np.array([self.meta_data[i][3] for i in _inds]).astype(int)
        
            poses = self.poses[_inds]
            mtds  = self.meta_data[_inds]
            clip_pose_dict[person_id] = [Pose(_pose,_mtd) for _pose,_mtd in  zip(poses,mtds)]
        
        frames = glob.glob(self.frame_dir + f"{clip_path}/*")
        return clip_gt, clip_pose_dict, frames

class Pose():
    def __init__(self,poses,mtd):
        self.dt = poses
        self.mtd = mtd
        
    def get_norm(self):
        dt = self.dt.copy()
        #for i in range(self.dt.shape[2]):
        #    dt[:,:,i]-=dt[:,:,i].min()
        #    dt[:,:,i]/=dt[:,:,i].max()
            
        for i in range(self.dt.shape[2]):
            dt[:,:,i]-=dt[:,:,i].min()
            dt[:,:,i]/=dt[:,:,i].std()
        return dt
    
    def get_polar(self):
        x = np.sqrt(self.dt[:,:,0]**2 + self.dt[:,:,1]**2)
        y = np.arctan2(self.dt[:,:,0],self.dt[:,:,1])
        return x,y
    
    def plot_pose(self,_id=list(range(18))):
        mtd=list(map(str,self.mtd))
        fname = f"{mtd[0].zfill(2)}_{mtd[1].zfill(4)}/{mtd[3].zfill(3)}.jpg"
        src = '/home/irfan/Desktop/Data/Pose_JSON_Data/ShanghaiTech/gt/test/frames/'+fname
        img = cv2.imread(src)
        fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(6,3))
        axes[0].imshow(img)
        for i in range(24):
            axes[1].scatter(self.dt[i,_id,0],self.dt[i,_id,1],marker='.')
        axes[1].imshow(img)
        plt.show()
        plt.close()
        
    def plot_hmap(self):
        
        dt     = self.get_norm()
        x,y    = self.get_polar()
        
        fig,axes = plt.subplots(nrows=1,ncols=dt.shape[2]+2,figsize=(10,3))
        for i in range(dt.shape[2]):
            axes[i].imshow(dt[:,:,i])
            
        axes[i+1].imshow(x)
        axes[i+2].imshow(y)
        for ax in axes:
            ax.set_xticks(range(11),rotation=90)
        plt.show()
        plt.close()
        
    def vis_data(self,_id = list(range(18))):
        __dt = self.dt.copy()
        __dt[:,:,0]  = self.dt[:,:,0] - self.dt[:,:,0].min()
        __dt[:,:,1]  = self.dt[:,:,1] - self.dt[:,:,1].min()
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
        
    
        
class Graph:

    def __init__(self,
                 layout='openpose',
                 strategy='spatial',
                 headless=False,
                 max_hop=1):
        self.headless = headless
        self.max_hop = max_hop
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A
    
    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 14
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1), (0, 1)]
            #neighbour_link = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
            #  [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17]]
            if not self.headless:
                neighbor_link += [(15, 0), (14, 0), (17, 15), (16, 14)]
                self.num_node = 18
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1

        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

