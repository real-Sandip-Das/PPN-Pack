# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""Data collection script."""

import argparse
import os
import json

import numpy as np

from ravens import Dataset
from ravens import Environment
from ravens import tasks
import pdb
import pybullet as p

from ppn_heuristic_packing import *
from geo_model import GeoBin
import trimesh

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_all_object_heightmaps(task, env):
    obs, reward, _, info = env.reset_packing(task)
    shape_codes = np.arange(96).tolist()
    #for i in np.arange(95,-1,-1).tolist():#range(96):
    for i in range(14):
        obj_type = shape_codes[i] # 69
        print ('item', i, 'loading type', obj_type)
        # Load the Mesh
        data_dir = os.path.join('.', 'autostore', 'models', 'scanned_model_dataset')
        obj_filename = os.path.join(data_dir, np.sort(os.listdir(data_dir))[obj_type])
        print (obj_filename)
        if not os.path.isfile(obj_filename):
            print ('Cannot find', obj_filename)
        obj = trimesh.load_mesh(obj_filename, file_type='ply', process=False)
        obj.export(os.path.join('.', 'autostore', 'models', 'scanned_model_dataset_obj', os.path.basename(obj_filename[:-4])+'.obj'))
        obj_top_down, obj_bottom_up, _ = get_high_res_object_heightmaps_and_occs(shape_codes, i, task, env, obj.bounds[1,2]/100.0, obj.bounds[0,2]/100.0)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(obj_top_down)
        plt.subplot(1,2,2)
        plt.imshow(obj_bottom_up)
        plt.show()
        np.save(os.path.join('.', 'autostore', 'models', 'high_res_real_obj', \
            os.path.basename(obj_filename[:-4]+'_top_down.npy')), obj_top_down)
        np.save(os.path.join('.', 'autostore', 'models', 'high_res_real_obj', \
            os.path.basename(obj_filename[:-4]+'_bottom_up.npy')), obj_bottom_up)

def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--disp', default=False, type=str2bool)
    parser.add_argument('--task', default='simulate-irregular') #block-insertion #'packing-irregular'
    parser.add_argument('--mode', default='test')
    parser.add_argument('--accelerate', default='network', choices=['network', 'plain'])
    parser.add_argument('--stable', default=True, type=str2bool)
    parser.add_argument('--method', default='sdf', choices=['dblf','hm','mta','sdf','random', 'first'])
    parser.add_argument('--n', default=10, type=int)
    parser.add_argument('--split', default=0, type=int)
    parser.add_argument('--case_cnt', default=500, type=int)
    parser.add_argument('--topk', default=50, type=int)
    parser.add_argument('--contr_seq', default=True, type=str2bool)
    parser.add_argument('--vol_dec', default=False, type=str2bool)
    parser.add_argument('--cuda', default=True, type=str2bool)
    parser.add_argument('--sdf_remain_terms', default='1234')
    parser.add_argument('--testing_shape_codes_file', default='sequential_testing_shape_codes_96_type_2000_80_num_rand.npy')
    parser.add_argument('--output_file', default='output.json')
    args = parser.parse_args()

    # Initialize environment and task.
    env = Environment(args.disp, hz=480)
    task = tasks.names[args.task]()
    task.mode = args.mode
    kernal = compute_sdf_kernal(args.cuda)

    # Initialize scripted oracle agent and dataset.
    dataset = Dataset(os.path.join('data', f'{args.task}-{task.mode}'))

    # Train seeds are even and test seeds are odd.
    seed = dataset.max_seed
    if seed < 0:
      seed = -1 if (task.mode == 'test') else -2
    
    # Collect training data from oracle demonstrations.
    print(f'Oracle demonstration: {dataset.n_episodes + 1}/{args.n}')
    episode, total_reward = [], 0
    seed += 2
    np.random.seed(seed)
    data_dir = os.path.join('.', 'autostore', 'models', 'our_oriented_dataset')
    occ_dir = os.path.join('.', 'autostore', 'models', 'our_oriented_occs')
    avg_success_rate = 0.0
    avg_success_cnt = 0.0
    avg_compactness = 0.0
    avg_pyramidality = 0.0
    avg_time = 0.0
    avg_volume = 0.0
    output = []
    for shape_code_idx in range(args.split*args.case_cnt, (args.split+1)*args.case_cnt): #[782, 628, 912, 499, 304, 477, 310, 820, 97, 605, 233, 951, 510]:
        start_ = time.time()
        print ()
        print ('Testing Packing Case', shape_code_idx)
        model = GeoBin(x_max=32, y_max=32, z_max=30, thickness=2)
        obs, reward, _, info = env.reset_packing(task)

        # Loading objects
        tot_obj_num = 80 #
        shape_codes = np.load(args.testing_shape_codes_file)[shape_code_idx][0:tot_obj_num] #np.load('shape_codes_96_type_500_num_rand.npy')[shape_code_idx][0:tot_obj_num] #np.load('shape_codes_5_type_80_num_rand.npy')[shape_code_idx][0:tot_obj_num]
        train_data_save_dict = []
        max_num_vertices = 0
        max_num_faces = 0
        dict_obj_occ = {}
        for i in range(len(shape_codes)):
            obj_type = shape_codes[i] # 69
            print ('item', i, 'loading type', obj_type)
            # Load the Mesh
            obj_filename = os.path.join(data_dir, np.sort(os.listdir(data_dir))[obj_type])
            print (obj_filename)
            if not os.path.isfile(obj_filename):
                print ('Cannot find', obj_filename)
            obj = trimesh.load_mesh(obj_filename, file_type='ply', process=False)
            model.add_object(obj)
            # Build the compose_occ for single objects
            for r in range(4):
                dict_obj_occ[str(i) + '_r' + str(r)] = np.load(os.path.join(occ_dir, os.path.basename(obj_filename)[:-4]+'_objocc.npy'), allow_pickle=True).item()['r' + str(r)]
        
        p.setGravity(0,0,-9.80665)
        list_placed = []
        if args.contr_seq:
             _,_,_,tot_time, packed_volumes, scene_top_down, train_data_save_dict = heuristic_ppn_packing(args, model, kernal, shape_codes, env, dict_obj_occ, require_stability=args.stable, choose_objective=args.method, use_cuda=args.cuda, sdf_remain_terms=args.sdf_remain_terms, vol_dec=args.vol_dec, train_data_save_dict=train_data_save_dict)
        else:
            raise NotImplementedError("Please set --contr_seq as True")
        success_cnt = 0
        packed_volume = 0.0
        i = -1
        for k in env.obj_ids['rigid']:
            i += 1
            if p.getBasePositionAndOrientation(k)[0][0] < 0.5-0.33/2 or p.getBasePositionAndOrientation(k)[0][0] > 0.5+0.33/2 or \
                p.getBasePositionAndOrientation(k)[0][1] < -0.33/2 or p.getBasePositionAndOrientation(k)[0][1] > 0.33/2 or \
                p.getBasePositionAndOrientation(k)[0][2] < 0 or p.getBasePositionAndOrientation(k)[0][2] > 0.30:
                success_cnt += 0
            else:
                success_cnt += 1
                packed_volume += packed_volumes[i]
        print ('Success Cnt', success_cnt)
        print ('Time Spent:', time.time() - start_)
        avg_success_cnt += success_cnt/args.case_cnt
        avg_success_rate += success_cnt/tot_obj_num/args.case_cnt
        avg_compactness += packed_volume/scene_top_down.max()/32/32*5/args.case_cnt
        avg_pyramidality += packed_volume/scene_top_down.sum()*5/args.case_cnt
        avg_time += tot_time/success_cnt/args.case_cnt
        avg_volume += packed_volume/args.case_cnt
        
        print ('avg_success_rate', avg_success_rate/(shape_code_idx+1-args.case_cnt*args.split)*args.case_cnt)
        print ('avg_compactness', avg_compactness/(shape_code_idx+1-args.case_cnt*args.split)*args.case_cnt)
        print ('avg_pyramidality',avg_pyramidality/(shape_code_idx+1-args.case_cnt*args.split)*args.case_cnt)
        print ('average_success_cnt', avg_success_cnt/(shape_code_idx+1-args.case_cnt*args.split)*args.case_cnt)
        print ('average_time_per_obj', avg_time/(shape_code_idx+1-args.case_cnt*args.split)*args.case_cnt)
        print ('average_volume', avg_volume/(shape_code_idx+1-args.case_cnt*args.split)*args.case_cnt)

        output.append([])
        for i, id in enumerate(env.obj_ids['rigid']):
            pos, orn = p.getBasePositionAndOrientation(id)
            output[-1].append({
                'pos': pos,
                'orn': orn,
                'shape_code': int(shape_codes[i])
            })
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=4)
    with open("shape_codes_to_filename.json", "w") as f:
        json.dump({i: np.sort(os.listdir(data_dir))[i] for i in range(len(np.sort(os.listdir(data_dir))))}, f, indent=4)

if __name__ == '__main__':
  main()

# TODO: modify environment.yml with current versions mentioned
