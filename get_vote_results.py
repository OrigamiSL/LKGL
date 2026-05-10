import argparse
import numpy as np
import os
from sample4geo.evaluate.university import compute_mAP, evaluate_SDM
import torch
parser = argparse.ArgumentParser(description='xx')
parser.add_argument('--itr', default=0, type=int)
parser.add_argument('--vote_path', default='./vote_results', type=str)
parser.add_argument('--dataset', default='U1652-D2S', type=str, help="'U1652-D2S' | 'U1652-S2D'")
parser.add_argument('--altitude', default=150, type=int, help="150|200|250|300|666, 666 is all data")
parser.add_argument('--dataset_name', default='SUES-200', type=str)

parser.add_argument('--result_path', default='vote_result.txt', type=str)
args = parser.parse_args()

if args.dataset_name == 'SUES-200':
    vote_path = os.path.join(args.vote_path, args.dataset_name, args.dataset, str(args.altitude))
else:
    vote_path = os.path.join(args.vote_path, args.dataset_name, args.dataset)

ql_path = os.path.join(vote_path, 'ql.npy')
gl_path = os.path.join(vote_path, 'gl.npy')
ql, gl = np.load(ql_path), np.load(gl_path)
ql_len = len(ql)

use_itr = [i+1 for i in range(10)]
# use_itr = [1, 2, 3, 6, 7, 8, 9, 10]

CMC = torch.IntTensor(len(gl)).zero_()
ap = 0.0
ranks=[1, 5, 10]
indexOfTopK_list = []

for q_idx in range(ql_len):
    score_all = 0
    for i in use_itr:
        score_path = os.path.join(vote_path, 'itr_'+str(i), 'score_'+str(i)+'_'+str(q_idx)+'.npy')
        score_i = np.load(score_path)
        score_all += score_i

    index = np.argsort(score_all / len(use_itr))  # from small to large
    index = index[::-1]

    # good index
    query_index = np.argwhere(gl == ql[q_idx])
    good_index = query_index

    # junk index
    junk_index = np.argwhere(gl == -1)

    CMC_tmp = compute_mAP(index, good_index, junk_index)

    ap_tmp, CMC_tmp, index_rank = CMC_tmp + (index, )

    indexOfTopK_list.append(index_rank)

    if CMC_tmp[0] == -1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp

AP = ap/ql_len*100
    
CMC = CMC.float()
CMC = CMC/ql_len #average CMC

# top 1%
top1 = round(len(gl)*0.01)
print('top1', top1)

string = []
            
for i in ranks:
    string.append('Recall@{}: {:.4f}'.format(i, CMC[i-1]*100))
    
string.append('Recall@top1: {:.4f}'.format(CMC[top1]*100))
string.append('AP: {:.4f}'.format(AP))   

# if args.dataset_name == 'DenseUAV':
#         path_denseUAV = '/home/lhg/work/ssd_new/visual_geolocalization/denseUAV/DenseUAV'
#         gps_file_path = os.path.join(path_denseUAV, "Dense_GPS_ALL.txt")
#         path_query, path_gallery = os.path.join(path_denseUAV,'test/query_drone'), os.path.join(path_denseUAV,'test/gallery_satellite')
        
#         if os.path.isfile(gps_file_path):
#             print("Loading GPS config for SDM/MA computation...")
#             configDict = {}
#             with open(gps_file_path, "r") as F:
#                 context = F.readlines()
#                 for line in context:
#                     splitLineList = line.strip().split(" ")
#                     configDict[splitLineList[0].split("/")[-2]] = [float(splitLineList[1].split("E")[-1]),
#                                                                 float(splitLineList[2].split("N")[-1])]
            
#             print("Computing SDM@K...")
#             SDM_dict = {}
#             for K_val in range(1, 101, 1):
#                 metric = 0
#                 for i in range(ql_len):
#                     P_ = evaluate_SDM(indexOfTopK_list[i], i, K_val, 
#                                       path_query, path_gallery, configDict)
#                     metric += P_
#                 metric = metric / ql_len
                
#                 if K_val in ranks:
#                     string.append("SDM@{} = {:.2f}%".format(K_val, metric * 100))
    
print(' - '.join(string)) 

with open (os.path.join(vote_path, 'vote_results.txt'), 'a') as f:
    f.write('*'*8 + 'vote_result' + '*'*8 + '\n')
    f.write('_'.join(string))
    f.write('\n')
    f.close()


