import os
import numpy as np

from .matching import orb_matching
from .utils import expand_bbox, stack_all_pids, best_matching_hungarian

def get_box(pose, img_height, img_width):
    xmin = pose[:,0].min()
    xmax = pose[:,0].max()
    ymin = pose[:,1].min()
    ymax = pose[:,1].max()
    return expand_bbox(xmin, xmax, ymin, ymax, img_width, img_height)

#The wrapper of PoseFlow algorithm to be embedded in alphapose inference
class PoseFlowWrapper():
    def __init__(self, link=25, drop=2.0, num=7,
                 mag=30, match=0.8, save_path='.tmp/poseflow', pool_size=5):
        # super parameters
        # 1. look-ahead LINK_LEN frames to find tracked human bbox
        # 2. bbox_IoU(deepmatching), bbox_IoU(general), pose_IoU(deepmatching), pose_IoU(general), box1_score, box2_score
        # 3. bbox_IoU(deepmatching), bbox_IoU(general), pose_IoU(deepmatching), pose_IoU(general), box1_score, box2_score(Non DeepMatching)
        # 4. drop low-score(<DROP) keypoints
        # 5. pick high-score(top NUM) keypoints when computing pose_IOU
        # 6. box width/height around keypoint for computing pose IoU
        # 7. match threshold in Hungarian Matching
        self.link_len    = link
        self.weights     = [1,2,1,2,0,0] 
        self.weights_fff = [0,1,0,1,0,0]
        self.drop = drop
        self.num  = num
        self.mag  = mag
        self.match_thres = match
        self.notrack = {}
        self.track = {}
        self.save_path = save_path
        self.save_match_path = os.path.join(save_path,'matching')
        self.pool_size = pool_size

        #if not os.path.exists(save_path):
        #    os.mkdir(save_path)

        #init local variables
        self.max_pid_id = 0
        self.prev_img = None
        print("Start pose tracking...\n")

    
    
    def _return(self,frame_id):
        for pid in range(self.track[frame_id]['num_boxes']):
            self.track[frame_id][pid]['new_pid'] = pid
            self.track[frame_id][pid]['match_score'] = 0
        return self.track[frame_id]#self.final_result_by_name(frame_id)

        
    def step(self, img, alphapose_results):
        frame_id   = alphapose_results["imgname"]  
        self.track[frame_id] = {}
        self.track[frame_id].update({pid:res for pid,res in enumerate(alphapose_results['result'])})
        self.track[frame_id].update({'num_boxes':len(alphapose_results['result'])})#self.convert_results_to_no_track(alphapose_results)
        
        # init tracking info of the first frame in one video
        if len(self.track.keys()) == 1:
            self.prev_img = img.copy()
            return self._return(frame_id)
        
        frame_id_list  = sorted([i for i in self.track.keys()])
        prev_frame_id  = frame_id_list[-2]
        frame_new_pids = []

        self.max_pid_id = max(self.max_pid_id, self.track[prev_frame_id]['num_boxes'])
        all_cors        = orb_matching(self.prev_img, img, prev_frame_id, frame_id)
        
        if self.track[frame_id]['num_boxes'] == 0:
            self.track[frame_id] = copy.deepcopy(self.track[prev_frame_id])
            self.prev_img        = img.copy()
            return self.track[frame_id]#self.final_result_by_name(frame_id)
        
        cur_all_pids, cur_all_pids_fff = stack_all_pids(self.track, 
                                                        frame_id_list,
                                                        self.link_len)
        
        match_indexes, match_scores    = best_matching_hungarian(all_cors,
                                                                 cur_all_pids,
                                                                 cur_all_pids_fff,
                                                                 self.track[frame_id],
                                                                 self.weights,
                                                                 self.weights_fff,
                                                                 self.num,
                                                                 self.mag,
                                                                 pool_size=self.pool_size)
        #if int(frame_id)==25 : import pdb;pdb.set_trace()
        for pid1, pid2 in match_indexes:
            if match_scores[pid1][pid2] > self.match_thres:
                self.track[frame_id][pid2]['new_pid']     = cur_all_pids[pid1]['new_pid']
                self.max_pid_id                           = max(self.max_pid_id, self.track[frame_id][pid2]['new_pid'])
                self.track[frame_id][pid2]['match_score'] = match_scores[pid1][pid2]

        # add the untracked new person
        for next_pid in range(self.track[frame_id]['num_boxes']):
            if 'new_pid' not in self.track[frame_id][next_pid]:
                self.max_pid_id += 1
                self.track[frame_id][next_pid]['new_pid'] = self.max_pid_id
                self.track[frame_id][next_pid]['match_score'] = 0

        self.prev_img = img.copy()
        return self.track[frame_id]#self.final_result_by_name(frame_id)







        