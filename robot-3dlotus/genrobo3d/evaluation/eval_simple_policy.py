from typing import Tuple, Dict, List

import os
import json
import jsonlines
import tap
import copy
from pathlib import Path
from filelock import FileLock

import torch
import numpy as np
from scipy.special import softmax

# TODO: error when import in a different order: Error /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.34’ not found or /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
# TODO: always import torch first
import open3d as o3d
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial.transform import Rotation as R

from genrobo3d.train.utils.misc import set_random_seed
from genrobo3d.configs.default import get_config

try:
    from genrobo3d.rlbench.environments import RLBenchEnv
except:
    print('No RLBench')

from genrobo3d.train.train_simple_policy import MODEL_FACTORY

from genrobo3d.configs.rlbench.constants import get_robot_workspace, get_rlbench_labels
from genrobo3d.utils.robot_box import RobotBox
from genrobo3d.train.datasets.common import gen_seq_masks
from genrobo3d.evaluation.common import write_to_file
from genrobo3d.utils.resize_input_rgb_pc import process_image, resize_point_cloud

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

import re
def split_instructions(s):
    # 使用正則表達式分割，處理逗號、then 及其組合
    parts = re.split(r'\s*(?:,+\s*then\s*|then|,)\s*', s)
    # 去除空格並過濾空元素
    return [part.strip() for part in parts if part.strip()]


class Arguments(tap.Tap):
    exp_config: str
    device: str = 'cuda'  # cpu, cuda

    microstep_data_dir: str = ''
    seed: int = 100  # seed for RLBench
    num_demos: int = 20
    taskvar: str = 'push_button+0'
    checkpoint: str = None

    headless: bool = False
    max_tries: int = 10
    max_steps: int = 25
    cam_rand_factor: float = 0.0
    image_size: List[int] = [256, 256]

    save_image: bool = False
    save_obs_outs_dir: str = None
    record_video: bool = False
    not_include_robot_cameras: bool = False
    video_rotate_cam: bool = False
    video_resolution: int = 480

    num_ensembles: int = 1

    best_disc_pos: str = 'max' # max, ens1

    real_robot: bool = False

class Actioner(object):
    def __init__(self, args) -> None:
        self.args = args
        if self.args.save_obs_outs_dir is not None:
            os.makedirs(self.args.save_obs_outs_dir, exist_ok=True)

        self.WORKSPACE = get_robot_workspace(real_robot=args.real_robot)
        self.device = torch.device(args.device)

        config = get_config(args.exp_config, args.remained_args)
        self.config = config
        self.config.defrost()
        self.config.TRAIN_DATASET.sample_points_by_distance = self.config.TRAIN_DATASET.get('sample_points_by_distance', False)
        self.config.TRAIN_DATASET.rm_pc_outliers = self.config.TRAIN_DATASET.get('rm_pc_outliers', False)
        self.config.TRAIN_DATASET.rm_pc_outliers_neighbors = self.config.TRAIN_DATASET.get('rm_pc_outliers_neighbors', 10)
        self.config.TRAIN_DATASET.same_npoints_per_example = self.config.TRAIN_DATASET.get('same_npoints_per_example', False)
        self.config.MODEL.action_config.best_disc_pos = args.best_disc_pos

        if args.checkpoint is not None:
            config.checkpoint = args.checkpoint

        model_class = MODEL_FACTORY[config.MODEL.model_class]
        self.model = model_class(config.MODEL)
        if config.checkpoint:
            checkpoint = torch.load(
                config.checkpoint, map_location=lambda storage, loc: storage
            )
            self.model.load_state_dict(checkpoint, strict=True)

        self.model.to(self.device)
        self.model.eval()

        self.config.freeze()

        data_cfg = self.config.TRAIN_DATASET
        self.data_cfg = data_cfg
        self.instr_embeds = np.load(data_cfg.instr_embed_file, allow_pickle=True).item()
        if data_cfg.instr_embed_type == 'last':
            self.instr_embeds = {instr: embeds[-1:] for instr, embeds in self.instr_embeds.items()}
        self.taskvar_instrs = json.load(open(data_cfg.taskvar_instr_file))

        self.TABLE_HEIGHT = self.WORKSPACE['TABLE_HEIGHT']
        
        
        ### my load agent ###
        from sam2act.eval import load_agent
        self.agent = load_agent(
            # model_path='SAM2Act/sam2act/runs/sam2act_rlbench/model_89.pth',               # e.g., self.model_path
            # model_path='SAM2Act/sam2act/runs/sam2act_rlbench/model_gembench_12epoch.pth', 
            model_path='SAM2Act/sam2act/runs/sam2act_rlbench/model_gembench_0epoch.pth',     
            # model_path='SAM2Act/sam2act/runs/sam2act_rlbench/model_special_26.pth',
            exp_cfg_path=None,            # e.g., self.exp_cfg_path
            mvt_cfg_path=None,            # e.g., self.mvt_cfg_path
            eval_log_dir='SAM2Act/sam2act/runs/sam2act_rlbench/eval/test/1',            # e.g., self.eval_log_dir
            device=0,                    # e.g., 整數，將被轉成 f"cuda:{device}"
            use_input_place_with_mean=False,
        )
        
        # 初始化 Gemma 3 LLM pipeline，用於拆分 instructions
        # 需要事先執行: huggingface-cli login

        # model_id = "google/gemma-3-1b-it"
        # splitter_model = AutoModelForCausalLM.from_pretrained(
        #     model_id,
        #     device_map="auto",
        #     torch_dtype=torch.bfloat16,
        #     use_auth_token=True,
        #     trust_remote_code=True       # 信任遠端程式碼
        # )
        # splitter_tokenizer = AutoTokenizer.from_pretrained(
        #     model_id,
        #     padding_side="left",
        #     truncation_side="left",
        #     trust_remote_code=True       # 同樣信任遠端程式碼
        # )
        # self.splitter = pipeline(
        #     "text-generation",
        #     model=splitter_model,
        #     tokenizer=splitter_tokenizer
        # )
        
        self._episode_length = 25
        self.init_gripper_pose_isSet = False
        self.init_gripper_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])

        ### my load agent ###

    def _get_mask_with_label_ids(self, sem, label_ids):
        mask = sem == label_ids[0]
        for label_id in label_ids[1:]:
            mask = mask | (sem == label_id)
        return mask
    
    def _get_mask_with_robot_box(self, xyz, arm_links_info, rm_robot_type):
        if rm_robot_type == 'box_keep_gripper':
            keep_gripper = True
        else:
            keep_gripper = False
        robot_box = RobotBox(
            arm_links_info, keep_gripper=keep_gripper, 
            env_name='real' if self.args.real_robot else 'rlbench'
        )
        _, robot_point_ids = robot_box.get_pc_overlap_ratio(xyz=xyz, return_indices=True)
        robot_point_ids = np.array(list(robot_point_ids))
        mask = np.ones((xyz.shape[0], ), dtype=bool)
        if len(robot_point_ids) > 0:
            mask[robot_point_ids] = False
        return mask
    
    def _rm_pc_outliers(self, xyz, rgb=None):
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(xyz)
        # pcd, idxs = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
        # pcd, idxs = pcd.remove_radius_outlier(nb_points=16, radius=0.03)
        clf = LocalOutlierFactor(n_neighbors=self.data_cfg.rm_pc_outliers_neighbors)
        preds = clf.fit_predict(xyz)
        idxs = (preds == 1)
        xyz = xyz[idxs]
        if rgb is not None:
            rgb = rgb[idxs]
        return xyz, rgb
    
    def process_point_clouds(
        self, xyz, rgb, gt_sem=None, ee_pose=None, arm_links_info=None, taskvar=None
    ):
        # keep points in robot workspace
        xyz = xyz.reshape(-1, 3)
        in_mask = (xyz[:, 0] > self.WORKSPACE['X_BBOX'][0]) & (xyz[:, 0] < self.WORKSPACE['X_BBOX'][1]) & \
                  (xyz[:, 1] > self.WORKSPACE['Y_BBOX'][0]) & (xyz[:, 1] < self.WORKSPACE['Y_BBOX'][1]) & \
                  (xyz[:, 2] > self.WORKSPACE['Z_BBOX'][0]) & (xyz[:, 2] < self.WORKSPACE['Z_BBOX'][1])
        if self.data_cfg.rm_table:
            in_mask = in_mask & (xyz[:, 2] > self.WORKSPACE['TABLE_HEIGHT'])
        xyz = xyz[in_mask]
        rgb = rgb.reshape(-1, 3)[in_mask]
        if gt_sem is not None:
            gt_sem = gt_sem.reshape(-1)[in_mask]

        # downsampling
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd, _, trace = pcd.voxel_down_sample_and_trace(
            self.config.MODEL.action_config.voxel_size, np.min(xyz, 0), np.max(xyz, 0)
        )
        xyz = np.asarray(pcd.points)
        trace = np.array([v[0] for v in trace])
        rgb = rgb[trace]
        if gt_sem is not None:
            gt_sem = gt_sem[trace]

        if self.args.real_robot:
            for _ in range(1):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz)
                pcd.colors = o3d.utility.Vector3dVector(rgb)
                pcd, outlier_masks = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.2)
                xyz = xyz[outlier_masks]
                rgb = rgb[outlier_masks]
                if gt_sem is not None:
                    gt_sem = gt_sem[outlier_masks]

        # remove non-object points
        if not self.args.real_robot:
            rm_label_ids = get_rlbench_labels(
                taskvar.split('+')[0], table=self.data_cfg.rm_table, robot=(self.data_cfg.rm_robot=='gt'), wall=False, floor=False
            )
            if len(rm_label_ids) > 0:
                rm_mask = self._get_mask_with_label_ids(gt_sem, rm_label_ids)
                xyz = xyz[~rm_mask]
                rgb = rgb[~rm_mask]
        
        if self.data_cfg.rm_robot.startswith('box'):
            mask = self._get_mask_with_robot_box(xyz, arm_links_info, self.data_cfg.rm_robot)
            xyz = xyz[mask]
            rgb = rgb[mask]

        if self.data_cfg.rm_pc_outliers:
            xyz, rgb = self._rm_pc_outliers(xyz, rgb)

        # sampling points
        if len(xyz) > self.data_cfg.num_points:
            if self.data_cfg.sample_points_by_distance:
                dists = np.sqrt(np.sum((xyz - ee_pose[:3])**2, 1))
                probs = 1 / np.maximum(dists, 0.1)
                probs = np.maximum(softmax(probs), 1e-30) 
                probs = probs / sum(probs)
                # probs = 1 / dists
                # probs = probs / np.sum(probs)
                point_idxs = np.random.choice(len(xyz), self.data_cfg.num_points, replace=False, p=probs)
            else:
                point_idxs = np.random.choice(len(xyz), self.data_cfg.num_points, replace=False)
        else:
            if self.data_cfg.same_npoints_per_example:
                point_idxs = np.random.choice(xyz.shape[0], self.data_cfg.num_points, replace=True)
            else:
                point_idxs = np.arange(xyz.shape[0])
        xyz = xyz[point_idxs]
        rgb = rgb[point_idxs]
        height = xyz[:, -1] - self.TABLE_HEIGHT

        # normalize
        if self.data_cfg.xyz_shift == 'none':
            centroid = np.zeros((3, ))
        elif self.data_cfg.xyz_shift == 'center':
            centroid = np.mean(xyz, 0)
        elif self.data_cfg.xyz_shift == 'gripper':
            centroid = copy.deepcopy(ee_pose[:3])
        if self.data_cfg.xyz_norm:
            radius = np.max(np.sqrt(np.sum((xyz - centroid) ** 2, axis=1)))
        else:
            radius = 1

        xyz = (xyz - centroid) / radius
        height = height / radius
        ee_pose[:3] = (ee_pose[:3] - centroid) / radius
        
        rgb = (rgb / 255.) * 2 - 1
        pc_ft = np.concatenate([xyz, rgb], 1)
        if self.data_cfg.get('use_height', False):
            pc_ft = np.concatenate([pc_ft, height[:, None]], 1)

        return pc_ft, centroid, radius, ee_pose


    def preprocess_obs(self, taskvar, step_id, obs):
        rgb = np.stack(obs['rgb'], 0)  # (N, H, W, C)
        xyz = np.stack(obs['pc'], 0)  # (N, H, W, C)
        if 'gt_mask' in obs:
            gt_sem = np.stack(obs['gt_mask'], 0)  # (N, H, W) 
        else:
            gt_sem = None
        
        # select one instruction
        instr = self.taskvar_instrs[taskvar][0]
        instr_embed = self.instr_embeds[instr]
        
        pc_ft, pc_centroid, pc_radius, ee_pose = self.process_point_clouds(
            xyz, rgb, gt_sem=gt_sem, ee_pose=copy.deepcopy(obs['gripper']), 
            arm_links_info=obs['arm_links_info'], taskvar=taskvar
        )
        
        batch = {
            'pc_fts': torch.from_numpy(pc_ft).float(),
            'pc_centroids': pc_centroid,
            'pc_radius': pc_radius,
            'ee_poses': torch.from_numpy(ee_pose).float().unsqueeze(0),
            'step_ids': torch.LongTensor([step_id]),
            'txt_embeds': torch.from_numpy(instr_embed).float(),
            'txt_lens': [instr_embed.shape[0]],
            'npoints_in_batch': [pc_ft.shape[0]],
            'offset': torch.LongTensor([pc_ft.shape[0]]),
        }
        if self.config.MODEL.model_class == 'SimplePolicyPCT':
            batch['pc_fts'] = batch['pc_fts'].unsqueeze(0)
            batch['txt_masks'] = torch.from_numpy(
                gen_seq_masks(batch['txt_lens'])
            ).bool()
            batch['txt_embeds'] = batch['txt_embeds'].unsqueeze(0)
            
        # for k, v in batch.items():
        #     if k not in ['pc_centroids', 'pc_radius', 'npoints_in_batch']:
        #         print(k, v.size())
        return batch

    # original code
    # def predict(
    #     self, task_str=None, variation=None, step_id=None, obs_state_dict=None, 
    #     episode_id=None, instructions=None,
    # ):
    #     # with open("/home/bill/Documents/research/CVPR_gembench_baseline/3dlotus.txt", "a") as file:
    #     #     print('hello7777')
    #     #     print('obs_state_dict keys = ', obs_state_dict.keys(), file=file)
    #     #     print('obs_state_dict rgb shape = ', obs_state_dict['rgb'].shape, file=file)
    #     #     print('obs_state_dict pc shape = ', obs_state_dict['pc'].shape, file=file)
    #     #     print('obs_state_dict = ', obs_state_dict, file=file)
    #     #     print('instructions = ', instructions, file=file)
        
    #     instructions = []
    #     taskvar = f'{task_str}+{variation}'
    #     batch = self.preprocess_obs(
    #         taskvar, step_id, obs_state_dict,
    #     )
    #     # print('heeoo')
    #     with torch.no_grad():
    #         actions = []
    #         # TODO
    #         for _ in range(self.args.num_ensembles):
    #             action = self.model(batch)[0].data.cpu()
    #             actions.append(action)
    #         if len(actions) > 1:
    #             # print(torch.stack(actions, 0))
    #             avg_action = torch.stack(actions, 0).mean(0)
    #             pred_rot = torch.from_numpy(R.from_euler(
    #                 'xyz', np.mean([R.from_quat(x[3:-1]).as_euler('xyz') for x in actions], 0),
    #             ).as_quat())
    #             action = torch.cat([avg_action[:3], pred_rot, avg_action[-1:]], 0)
    #         else:
    #             action = actions[0]
    #     action[-1] = torch.sigmoid(action[-1]) > 0.5
        
    #     # action = action.data.cpu().numpy()
    #     action = action.numpy()
    #     action[:3] = action[:3] * batch['pc_radius'] + batch['pc_centroids']
    #     # TODO: ensure the action height is above the table
    #     action[2] = max(action[2], self.TABLE_HEIGHT+0.005)

    #     out = {
    #         'action': action
    #     }

    #     if self.args.save_obs_outs_dir is not None:
    #         np.save(
    #             os.path.join(self.args.save_obs_outs_dir, f'{task_str}+{variation}-{episode_id}-{step_id}.npy'),
    #             {
    #                 'batch': {k: v.data.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in batch.items()},
    #                 'obs': obs_state_dict,
    #                 'action': action
    #             }
    #         )
    #     # with open("/home/bill/Documents/research/CVPR_gembench_baseline/3dlotus.txt", "a") as file:
    #     #     print('out type = ', type(out), file=file)
    #     #     print('out keys = ', out.keys(), file=file)
    #     #     print('out action = ', out['action'], file=file)
    #     #     print('out action shape = ', out['action'].shape, file=file)
    #     #     print('out action type = ', type(out['action']), file=file)
    #     #     print('out = ', out, file=file)
    #     #     print("~~~~~~~~~~~~~~~~~~")
    #     return out
    
    # def predict(
    #     self, task_str=None, variation=None, step_id=None, obs_state_dict=None, 
    #     episode_id=None, instructions=None,
    # ):
    #     """
    #     根據輸入的 obs_state_dict，將資料拆解成 4 張影像／點雲，再組成 policy 所需的觀測輸入，
    #     然後呼叫 agent.act 得到 ActResult，最後擷取前 8 個 action 元素，並回傳 dict 格式。
        
    #     輸入：
    #     - obs_state_dict: dict，包含 keys ['rgb', 'depth', 'pc', 'arm_links_info', 'gt_mask', 'gripper']
    #         * rgb shape: (4, 256, 256, 3)
    #         * pc shape: (4, 256, 256, 3)
    #         * gripper: 8 維向量，其中第 8 個數值是 gripper 的開合狀態（但此處我們不用直接用）
        
    #     policy 觀測輸入的 keys：
    #     'left_shoulder_rgb', 'left_shoulder_point_cloud',
    #     'right_shoulder_rgb', 'right_shoulder_point_cloud',
    #     'wrist_rgb', 'wrist_point_cloud',
    #     'front_rgb', 'front_point_cloud',
    #     'ignore_collisions', 'low_dim_state', 'lang_goal_tokens'
        
    #     輸出：
    #     - dict，包含 'action'，其 value 為 shape (8,) 的 numpy array
    #     """
    #     # 將 4 張 rgb 和 4 張 point cloud 分別拆解
    #     rgb = obs_state_dict.get("rgb")
    #     pc = obs_state_dict.get("pc")
        
    #     # 檢查資料完整性
    #     if rgb is None or pc is None:
    #         raise ValueError("obs_state_dict must contain 'rgb' and 'pc' keys")
    #     if rgb.shape[0] != 4 or pc.shape[0] != 4:
    #         raise ValueError("Expected 4 images for 'rgb' and 4 for 'pc'")
        
    #     # 設定縮放比例, 256*0.5 = 128
    #     scale = 0.5

    #     policy_obs = {
    #         "left_shoulder_rgb": [process_image(rgb[0], scale)],
    #         "left_shoulder_point_cloud": [resize_point_cloud(pc[0], scale)],
    #         "right_shoulder_rgb": [process_image(rgb[1], scale)],
    #         "right_shoulder_point_cloud": [resize_point_cloud(pc[1], scale)],
    #         "wrist_rgb": [process_image(rgb[2], scale)],
    #         "wrist_point_cloud": [resize_point_cloud(pc[2], scale)],
    #         "front_rgb": [process_image(rgb[3], scale)],
    #         "front_point_cloud": [resize_point_cloud(pc[3], scale)],
    #         "ignore_collisions": [[0.]]
    #     }
                
  
        
    #     # 3. 由 gripper 取得 low_dim_state
    #     # gripper: 8 維向量，最後一個數值代表 gripper_open (1 或 0)
    #     gripper_input = obs_state_dict.get("gripper")
    #     if gripper_input is None:
    #         raise ValueError("obs_state_dict 缺少 'gripper' 資料")
    #     gripper_open = gripper_input[-1]
    #     # 計算 time 參數，假設 self._episode_length 為總步數
    #     time_param = (1. - (step_id / float(25 - 1))) * 2 - 1.
    #     # 組成 low_dim_state
    #     low_dim_state = [np.array([gripper_open, 0.0, 0.0, time_param], dtype=np.float32)]
    #     policy_obs["low_dim_state"] = low_dim_state


    #     # 4. 取得 lang_goal_tokens，利用 clip.tokenize 處理 instructions
    #     original_instructions = None
    #     selected_instruction = None
    #     if isinstance(instructions, list):
    #         original_instructions = instructions.copy()
    #         # TODO: simply use the first one (might be better to use the longest one)
    #         selected_instruction = instructions[0]
    #         # Choose the instruction string with the maximum length
    #         # selected_instruction = max(instructions, key=len)
    #         # all concrecated instructions
    #         # selected_instruction = ", ".join(instructions)
            
    #     # 將 log 寫入指定資料夾（使用 JSON 格式）
    #     log_base_dir = os.path.join("data", "experiments", "gembench", "3dlotus", "v1", "preds", "instruction_log")
    #     os.makedirs(log_base_dir, exist_ok=True)
    #     log_file_path = os.path.join(log_base_dir, f"{task_str}+{variation}.json")

    #     # 若檔案存在則讀取，否則初始化空資料
    #     if os.path.exists(log_file_path):
    #         with open(log_file_path, "r", encoding="utf-8") as f:
    #             log_data = json.load(f)
    #     else:
    #         log_data = {
    #             "task_str": task_str,
    #             "variation": variation,
    #             "original_instructions": original_instructions,
    #             "steps": {}
    #         }

    #     # 更新該 step_id 對應的選擇 instruction
    #     log_data["steps"][str(step_id)] = {
    #         "selected_instruction": selected_instruction
    #     }

    #     # 寫回 JSON 檔案
    #     with open(log_file_path, "w", encoding="utf-8") as f:
    #         json.dump(log_data, f, indent=2, ensure_ascii=False)
            
    #     from clip import tokenize
    #     if selected_instruction is not None:
    #         lang_goal_tokens = tokenize([selected_instruction])
    #     else:
    #         lang_goal_tokens = np.array([])
    #     policy_obs["lang_goal_tokens"] = lang_goal_tokens
        
        
    #     # 呼叫 policy 函式 act，load agent
    #     # from sam2act.eval import load_agent
    #     # agent = load_agent(
    #     #     model_path='SAM2Act/sam2act/runs/sam2act_rlbench/model_89.pth',                # e.g., self.model_path
    #     #     exp_cfg_path=None,            # e.g., self.exp_cfg_path
    #     #     mvt_cfg_path=None,            # e.g., self.mvt_cfg_path
    #     #     eval_log_dir='SAM2Act/sam2act/runs/sam2act_rlbench/eval/test/1',            # e.g., self.eval_log_dir
    #     #     device=0,                    # e.g., 整數，將被轉成 f"cuda:{device}"
    #     #     use_input_place_with_mean=False,
    #     # )
        
       
    #     prepped_data = {k: torch.tensor(np.array([v]), device=self.device) for k, v in policy_obs.items()}
    #     act_result = self.agent.act(step=step_id, observation=prepped_data, deterministic=True)
        
    #     # 擷取 act_result.action 前 8 個資訊（預期 act_result.action 為 numpy array）
    #     full_action = act_result.action  # 例如 shape 為 (N,) 的 numpy array
    #     action = full_action[:8]
    #     # action[2] = max(action[2], self.TABLE_HEIGHT+0.005)
    #     # print("action = ", action)
    #     # 回傳 predict 的格式
    #     return {"action": action}
    
    # def predict(
    #     self, task_str=None, variation=None, step_id=None, obs_state_dict=None, 
    #     episode_id=None, instructions=None,
    # ):
    #     """
    #     透過 LLM 拆分指令成子任務，並根據 step_id 分配對應的子指令；
    #     同時記錄原始與子指令對應資訊至 JSON 檔。
    #     """
    #     # 1. 基本檢查與影像／點雲前處理
    #     rgb = obs_state_dict.get("rgb")
    #     pc = obs_state_dict.get("pc")
    #     if rgb is None or pc is None or rgb.shape[0] != 4 or pc.shape[0] != 4:
    #         raise ValueError("obs_state_dict must contain 4 'rgb' and 4 'pc' images")

    #     scale = 0.5
    #     policy_obs = {
    #         "left_shoulder_rgb": [process_image(rgb[0], scale)],
    #         "left_shoulder_point_cloud": [resize_point_cloud(pc[0], scale)],
    #         "right_shoulder_rgb": [process_image(rgb[1], scale)],
    #         "right_shoulder_point_cloud": [resize_point_cloud(pc[1], scale)],
    #         "wrist_rgb": [process_image(rgb[2], scale)],
    #         "wrist_point_cloud": [resize_point_cloud(pc[2], scale)],
    #         "front_rgb": [process_image(rgb[3], scale)],
    #         "front_point_cloud": [resize_point_cloud(pc[3], scale)],
    #         "ignore_collisions": [[0.]]
    #     }

    #     # 2. low_dim_state
    #     gripper_input = obs_state_dict.get("gripper")
    #     if gripper_input is None:
    #         raise ValueError("obs_state_dict 缺少 'gripper' 資料")
    #     gripper_open = gripper_input[-1]
    #     time_param = (1. - (step_id / float(self._episode_length - 1))) * 2 - 1.
    #     policy_obs["low_dim_state"] = [np.array([gripper_open, 0.0, 0.0, time_param], dtype=np.float32)]

    #     # 3. 處理原始 instructions
    #     if instructions is None:
    #         original_instructions = []
    #     elif isinstance(instructions, list):
    #         original_instructions = instructions.copy()
    #     else:
    #         original_instructions = [instructions]

    #     # 4. 使用 LLM 拆分成子指令
    #     #    我們要求 LLM 回傳 {"subinstructions": [...], "num_subinstructions": N}
    #     prompt = (
    #         "Split the following list of semantically equivalent robot instructions into "
    #         "atomic actions. Return a JSON object with two fields:\n"
    #         "1) subinstructions: an array of strings, each a single action;\n"
    #         "2) num_subinstructions: the total number of actions.\n\n"
    #         f"Instructions: {json.dumps(original_instructions, ensure_ascii=False)}"
    #     )
    #     # 呼叫 pipeline
    #     llm_out = self.splitter(
    #         prompt,
    #         max_new_tokens=128,
    #         do_sample=False
    #     )
    #     generated = llm_out[0]["generated_text"]
    #     fallback_used = False
    #     try:
    #         # 移除 prompt 文字，留下 JSON
    #         json_part = generated.replace(prompt, "", 1).strip()
    #         result = json.loads(json_part)
    #         subinstructions = result["subinstructions"]
    #         num_sub = int(result["num_subinstructions"])
    #         if not isinstance(subinstructions, list) or num_sub != len(subinstructions):
    #             raise ValueError("Invalid structure or count mismatch")
    #     except Exception:
    #         # fallback: 按行分割
    #         fallback_used = True
    #         lines = generated.splitlines()
    #         subinstructions = [ln.strip("- ").strip() for ln in lines if ln.strip()]
    #         num_sub = len(subinstructions)

    #     # 5. 若只有一個子指令，複製滿整個 episode
    #     if num_sub <= 1:
    #         num_sub = 1
    #         subinstructions = subinstructions or [""]
    #         subinstructions = [subinstructions[0]] * self._episode_length

    #     # 6. 計算每個子指令對應的步數區間
    #     steps_per = float(self._episode_length) / num_sub
    #     idx = min(int(step_id // steps_per), num_sub - 1)
    #     selected_instruction = subinstructions[idx]

    #     # 5. 紀錄 JSON 檔
    #     log_dir = os.path.join(
    #         "data", "experiments", "gembench", "3dlotus", "v1", "preds", "instruction_log"
    #     )
    #     os.makedirs(log_dir, exist_ok=True)
    #     log_path = os.path.join(log_dir, f"{task_str}+{variation}.json")
    #     # 讀取或初始化
    #     if os.path.exists(log_path):
    #         with open(log_path, 'r', encoding='utf-8') as f:
    #             log = json.load(f)
    #     else:
    #         log = {
    #             "task_str": task_str,
    #             "variation": variation,
    #             "original_instructions": original_instructions,
    #             "subinstructions": subinstructions,
    #             "fallback_used": fallback_used,
    #             "steps": {}
    #         }
    #     # 更新此 step
    #     log["steps"][str(step_id)] = {"selected": selected_instruction}
    #     with open(log_path, 'w', encoding='utf-8') as f:
    #         json.dump(log, f, ensure_ascii=False, indent=2)

    #     from clip import tokenize
    #     # 6. CLIP tokenize
    #     if selected_instruction:
    #         lang_goal_tokens = tokenize([selected_instruction])
    #     else:
    #         lang_goal_tokens = np.array([])
    #     policy_obs["lang_goal_tokens"] = lang_goal_tokens

    #     # 7. 呼叫 agent
    #     prepped = {k: torch.tensor(np.array([v]), device=self.device) for k, v in policy_obs.items()}
    #     act_result = self.agent.act(step=step_id, observation=prepped, deterministic=True)
    #     action = act_result.action[:8]
    #     return {"action": action}
    
    def predict(
        self, task_str=None, variation=None, step_id=None, obs_state_dict=None, 
        episode_id=None, instructions=None,
    ):
        """
        根據輸入的 obs_state_dict，將資料拆解成 4 張影像／點雲，再組成 policy 所需的觀測輸入，
        然後呼叫 agent.act 得到 ActResult，最後擷取前 8 個 action 元素，並回傳 dict 格式。
        
        輸入：
        - obs_state_dict: dict，包含 keys ['rgb', 'depth', 'pc', 'arm_links_info', 'gt_mask', 'gripper']
            * rgb shape: (4, 256, 256, 3)
            * pc shape: (4, 256, 256, 3)
            * gripper: 8 維向量，其中第 8 個數值是 gripper 的開合狀態（但此處我們不用直接用）
        
        policy 觀測輸入的 keys：
        'left_shoulder_rgb', 'left_shoulder_point_cloud',
        'right_shoulder_rgb', 'right_shoulder_point_cloud',
        'wrist_rgb', 'wrist_point_cloud',
        'front_rgb', 'front_point_cloud',
        'ignore_collisions', 'low_dim_state', 'lang_goal_tokens'
        
        輸出：
        - dict，包含 'action'，其 value 為 shape (8,) 的 numpy array
        """
        # 將 4 張 rgb 和 4 張 point cloud 分別拆解
        rgb = obs_state_dict.get("rgb")
        pc = obs_state_dict.get("pc")
        
        # 檢查資料完整性
        if rgb is None or pc is None:
            raise ValueError("obs_state_dict must contain 'rgb' and 'pc' keys")
        if rgb.shape[0] != 4 or pc.shape[0] != 4:
            raise ValueError("Expected 4 images for 'rgb' and 4 for 'pc'")
        
        # 設定縮放比例, 256*0.5 = 128
        scale = 0.5

        policy_obs = {
            "left_shoulder_rgb": [process_image(rgb[0], scale)],
            "left_shoulder_point_cloud": [resize_point_cloud(pc[0], scale)],
            "right_shoulder_rgb": [process_image(rgb[1], scale)],
            "right_shoulder_point_cloud": [resize_point_cloud(pc[1], scale)],
            "wrist_rgb": [process_image(rgb[2], scale)],
            "wrist_point_cloud": [resize_point_cloud(pc[2], scale)],
            "front_rgb": [process_image(rgb[3], scale)],
            "front_point_cloud": [resize_point_cloud(pc[3], scale)],
            "ignore_collisions": [np.array([0.], dtype=np.float32)]
        }
                
  
        
        # 3. 由 gripper 取得 low_dim_state
        # gripper: 8 維向量，最後一個數值代表 gripper_open (1 或 0)
        gripper_input = obs_state_dict.get("gripper")
        if gripper_input is None:
            raise ValueError("obs_state_dict 缺少 'gripper' 資料")
        gripper_open = gripper_input[-1]
        # 計算 time 參數，假設 self._episode_length 為總步數
        time_param = (1. - (step_id / float(self._episode_length - 1))) * 2 - 1.
        # if step_id == 0 or step_id == 1:
        #     time_param = 1.0
            
        
        # 組成 low_dim_state
        low_dim_state = [np.array([gripper_open, 0.0, 0.0, time_param], dtype=np.float32)]
        policy_obs["low_dim_state"] = low_dim_state
        
        if not self.init_gripper_pose_isSet:
            self.init_gripper_pose_isSet = True
            self.init_gripper_pose = gripper_input.copy()

        # 4. 取得 lang_goal_tokens，利用 clip.tokenize 處理 instructions
        original_instructions = None
        filter_instruction = None
        if isinstance(instructions, list):
            original_instructions = instructions.copy()
            # TODO: simply use the first one (might be better to use the longest one)
            filter_instruction = instructions[0]
            # Choose the instruction string with the maximum length
            # selected_instruction = max(instructions, key=len)
            # all concrecated instructions
            # selected_instruction = ", ".join(instructions)
        
        split_instruction_list = split_instructions(filter_instruction)
        # print("split_instruction_list = ", split_instruction_list)
        # print("self._episode_length = ", self._episode_length)
        # print("step_id = ", step_id)
        # print("index = ", int(step_id / (self._episode_length/len(split_instruction_list))))
        # stepNumPerSubtask = 3
        splitInstructionRepeatCount = 1
        stepNumPerSubtask = int(self._episode_length/len(split_instruction_list)/splitInstructionRepeatCount)
        selected_instruction = split_instruction_list[ int((step_id/stepNumPerSubtask)%len(split_instruction_list)) ]
            
        # 將 log 寫入指定資料夾（使用 JSON 格式）
        log_base_dir = os.path.join("data", "experiments", "gembench", "3dlotus", "v1", "preds", "instruction_log")
        os.makedirs(log_base_dir, exist_ok=True)
        log_file_path = os.path.join(log_base_dir, f"{task_str}+{variation}.json")

        # 若檔案存在則讀取，否則初始化空資料
        if os.path.exists(log_file_path):
            with open(log_file_path, "r", encoding="utf-8") as f:
                log_data = json.load(f)
        else:
            log_data = {
                "task_str": task_str,
                "variation": variation,
                "init_gripper_pose": self.init_gripper_pose.tolist(),
                "original_instructions": original_instructions,
                "steps": {}
            }

        # 更新該 step_id 對應的選擇 instruction
        log_data["steps"][str(step_id)] = {
            "selected_instruction": selected_instruction
        }

        # 寫回 JSON 檔案
        with open(log_file_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
            
        from clip import tokenize
        if selected_instruction is not None:
            lang_goal_tokens = tokenize([selected_instruction])
        else:
            lang_goal_tokens = np.array([])
        policy_obs["lang_goal_tokens"] = lang_goal_tokens
        
        
        # 呼叫 policy 函式 act，load agent
        # from sam2act.eval import load_agent
        # agent = load_agent(
        #     model_path='SAM2Act/sam2act/runs/sam2act_rlbench/model_89.pth',                # e.g., self.model_path
        #     exp_cfg_path=None,            # e.g., self.exp_cfg_path
        #     mvt_cfg_path=None,            # e.g., self.mvt_cfg_path
        #     eval_log_dir='SAM2Act/sam2act/runs/sam2act_rlbench/eval/test/1',            # e.g., self.eval_log_dir
        #     device=0,                    # e.g., 整數，將被轉成 f"cuda:{device}"
        #     use_input_place_with_mean=False,
        # )
        
       
        prepped_data = {k: torch.tensor(np.array([v]), device=self.device) for k, v in policy_obs.items()}
        act_result = self.agent.act(step=step_id, observation=prepped_data, deterministic=True)
        
        # 擷取 act_result.action 前 8 個資訊（預期 act_result.action 為 numpy array）
        full_action = act_result.action  # 例如 shape 為 (N,) 的 numpy array
        action = full_action[:8]
        # action[2] = max(action[2], self.TABLE_HEIGHT+0.005)
        # print(task_str, variation, episode_id, 'previous action = ', gripper_input)
        # print(task_str, variation, episode_id, "action = ", action)
        # print("~~~~~~")
        # 回傳 predict 的格式
        
        ####### initial pose #########
        if (step_id+1) % stepNumPerSubtask == 0: # because 0-indexed, step_id+1 == 1, 2, 3, ...
            print(task_str, variation, episode_id, f"step'{step_id} init!!")
            action = self.init_gripper_pose # return to initial pose
        ####### initial pose #########
        
        taskvar = f'{task_str}+{variation}'
        batch = self.preprocess_obs(
            taskvar, step_id, obs_state_dict,
        )
        
        if self.args.save_obs_outs_dir is not None:
            # 1. 組出 task+variation 這層資料夾
            task_dir = os.path.join(
                self.args.save_obs_outs_dir,
                f"{task_str}+{variation}"
            )
            # 2. 再組出 episode_id 這層資料夾
            episode_dir = os.path.join(task_dir, str(episode_id))

            # 3. 確保資料夾存在（exist_ok=True 不會在已存在時丟錯）
            os.makedirs(episode_dir, exist_ok=True)

            # 4. 最後用 step_id 作為檔名儲存
            file_name = f"{step_id}.npy"
            save_path = os.path.join(episode_dir, file_name)

            # 5. 儲存
            np.save(
                save_path,
                {
                    'batch': {
                        k: v.data.cpu().numpy() if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    },
                    'obs': obs_state_dict,
                    'action': action
                }
            )
        # print('action = ', action)
        # print('action type = ', type(action))
        # print('action[0] type = ', type(action[0]))
        # action = [0.20725445449352264,
        #     -0.3063616156578064,
        #     0.8873884081840515,
        #     -0.21263110997159382,
        #     -0.6743797232066278,
        #     0.6743797232066279,
        #     0.21263110997159385,
        #     0.0]
        return {"action": action}
    



def evaluate_actioner(args):    
    
    set_random_seed(args.seed)

    actioner = Actioner(args)
    
    pred_dir = os.path.join(actioner.config.output_dir, 'preds', f'seed{args.seed}')
    if args.cam_rand_factor > 0:
        pred_dir = '%s-cam_rand_factor%.1f' % (pred_dir, args.cam_rand_factor)
    os.makedirs(pred_dir, exist_ok=True)

    if len(args.image_size) == 1:
        args.image_size = [args.image_size[0], args.image_size[0]]    # (height, width)

    outfile = os.path.join(pred_dir, 'results.jsonl')

    existed_data = set()
    if os.path.exists(outfile):
        with jsonlines.open(outfile, 'r') as f:
            for item in f:
                existed_data.add((item['checkpoint'], '%s+%d'%(item['task'], item['variation'])))

    if (args.checkpoint, args.taskvar) in existed_data:
        return

    env = RLBenchEnv(
        data_path=args.microstep_data_dir,
        apply_rgb=True,
        apply_pc=True,
        apply_mask=True,
        headless=args.headless,
        image_size=args.image_size,
        cam_rand_factor=args.cam_rand_factor,
    )

    task_str, variation = args.taskvar.split('+')
    variation = int(variation)

    if args.microstep_data_dir != '':
        episodes_dir = os.path.join(args.microstep_data_dir, task_str, f"variation{variation}", "episodes")
        demo_keys, demos = [], []
        if os.path.exists(str(episodes_dir)):
            episode_ids = os.listdir(episodes_dir)
            episode_ids.sort(key=lambda ep: int(ep[7:]))
            for idx, ep in enumerate(episode_ids):
                # episode_id = int(ep[7:])
                try:
                    demo = env.get_demo(task_str, variation, idx, load_images=False)
                    demo_keys.append(f'episode{idx}')
                    demos.append(demo)
                except Exception as e:
                    print('\tProblem to load demo_id:', idx, ep)
                    print(e)
    else:
        demo_keys = None
        demos = None
            
    success_rate = env.evaluate(
        task_str, variation,
        actioner=actioner,
        max_episodes=args.max_steps,
        num_demos=len(demos) if demos is not None else args.num_demos,
        demos=demos,
        demo_keys=demo_keys,
        log_dir=Path(pred_dir),
        max_tries=args.max_tries,
        save_image=args.save_image,
        record_video=args.record_video,
        include_robot_cameras=(not args.not_include_robot_cameras),
        video_rotate_cam=args.video_rotate_cam,
        video_resolution=args.video_resolution,
    )

    print("Testing Success Rate {}: {:.04f}".format(task_str, success_rate))
    write_to_file(
        outfile,
        {
            'checkpoint': args.checkpoint,
            'task': task_str, 'variation': variation,
            'num_demos': args.num_demos, 'sr': success_rate
        }
    )



if __name__ == '__main__':
    args = Arguments().parse_args(known_only=True)
    args.remained_args = args.extra_args
    
    evaluate_actioner(args)
