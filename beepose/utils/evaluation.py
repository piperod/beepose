import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

import json
import glob,os
import cv2
from scipy.optimize import linear_sum_assignment as hungarian

def evaluator(gtr,det_full, parts_name={0:'Tail',1:'Head',2:'Torax',3:'Rant',4:'Lant'},threshold=50,scale=1):
    # Ground truth as an array
    nparts=len(parts_name.keys())
    ground = {}
    ground_mappings = {}
    for fr in gtr['annotations']:
        ground[fr['image_id']] = []
        ground_mappings[fr['image_id']] = []
    for fr in gtr['annotations']:
        if nparts ==3:
            ground_mappings[fr['image_id']].append([fr['keypoints'][:2],fr['keypoints'][4:6],
                                       distance(fr['keypoints'][:2],fr['keypoints'][2:4])])
        else:
            ground_mappings[fr['image_id']].append([fr['keypoints'][:2],fr['keypoints'][2:4],
                                       distance(fr['keypoints'][:2],fr['keypoints'][2:4])])
        
            
        for k in range(0,nparts*2-1,2):
            ground[fr['image_id']].append( [fr['keypoints'][k],fr['keypoints'][k+1]])

    # Detection keys as in ground truth
    detections = {}
    mappings = {}
    for key in det_full.keys():
        if key =='runningtime':
            continue
        detections[int(key.split("/")[-1].split(".")[0])]= np.array(det_full[key]['detections'])*scale
        temp = det_full[key]['mapping']
        mappings[int(key.split("/")[-1].split(".")[0])]= []
        for t in temp:
            mappings[int(key.split("/")[-1].split(".")[0])].append([np.array([t[1][0],t[0][0]])*scale,np.array([t[1][1],t[0][1]])*scale,t[2],t[3]])
    # Evaluations
    evaluations={}
    retrieval={}
    
    detect=detections
    for k in ground.keys():
        if k not in detect.keys():
            continue
        evaluations[k]={}
        retrieval[k]={}
        gt = ground[k]
        gt_map = ground_mappings[k]
        dt_map = mappings[k]
        dt = detect[k]
        cm = cost_matrix(gt,dt,threshold)
        gt_idx,dt_idx=hungarian(cm)
        cm_map=cost_matrix_mappings(ground_mappings[k],mappings[k],threshold)
        gmap_idx, dmap_idx=hungarian(cm_map)
        assignments_cost=cm[gt_idx,dt_idx]
        assignments_cost_map=cm_map[gmap_idx,dmap_idx]
        item=evaluations[k]
        item['gt_total_parts']=len(gt)
        item['gt_total_maps']=len(gt_map)
        item['dt_total_parts']=len(dt)
        item['dt_total_maps']=len(dt_map)
        item['gt_individuals']=len(gt)//nparts
        item['dt_individuals']=len(dt)//nparts
        item['cumulative_error']=cm[gt_idx[:len(gt)],dt_idx[:len(gt)]].sum()
        item['total_avg_error']=cm[gt_idx[:len(gt)],dt_idx[:len(gt)]].mean()
        item['total_std_error']=cm[gt_idx[:len(gt)],dt_idx[:len(gt)]].std()
        item['cumulative_error_map']=cm[gmap_idx[:len(gt_map)],dmap_idx[:len(gt_map)]].sum()
        item['avg_error_map']=cm[gmap_idx[:len(gt_map)],dmap_idx[:len(gt_map)]].mean()
        item['std_error_map']=cm[gmap_idx[:len(gt_map)],dmap_idx[:len(gt_map)]].std()
        #retrieval[k]['cost_matrix'] =cm
        #retrieval[k]['cost_matrix_map'] =cm_map
        #retrieval[k]['cost_matrix_assignments'] = assignments_cost
        #retrieval[k]['cost_matrix_assignments_map'] = assignments_cost_map
                                                      
        retrieval[k]['dt_idx']=dt_idx
        for i in range(nparts):
            temp=np.array([1 if assignments_cost[j]<threshold else 0 for j in  range(i,len(gt),nparts)])
            item[parts_name[i]+' score'] = 100*temp.sum()/evaluations[k]['gt_individuals']
            item[parts_name[i]+' total'] = temp.sum()

        
        temp=np.array([1 if assignments_cost_map[j]<threshold*2 else 0 for j in  range(len(gt_map))])
        item['matching_score'] = 100*temp.sum()/len(gt_map)
        item['matching_total'] = temp.sum()
    retrieval['ground']=ground
    retrieval['ground_mappings']=ground_mappings
    retrieval['detections'] = detections
    retrieval['mappings'] = mappings
    return evaluations,retrieval
        
def distance(dt,gt,tl=1):
    return ((dt[0]-gt[0])**2+(dt[1]-gt[1])**2)/tl

def distance_tracks(dt,gt,tl=1):
    return (np.sqrt((dt[0]-gt[0])**2+(dt[1]-gt[1])**2))/tl

def cost_matrix_tracks(ground_t,detections,threshold):
    total = len(ground_t)+len(detections)
    cost_m = np.zeros((total,total))
    for i in range(total):
        for j in range(total):
            if i < len(ground_t) and j <len(detections):
                cost_m[i][j] = distance_tracks(ground_t[i],detections[j])
            else:
                cost_m[i][j] = threshold
    
    return cost_m

def cost_matrix(ground_t,detections,threshold):
    total = len(ground_t)+len(detections)
    cost_m = np.zeros((total,total))
    for i in range(total):
        for j in range(total):
            if i < len(ground_t) and j <len(detections):
                cost_m[i][j] = distance(ground_t[i],detections[j],len(ground_t))
            else:
                cost_m[i][j] = threshold
    
    return cost_m

def cost_matrix_mappings(ground_mappings,mapings,threshold):
    total = len(ground_mappings)+len(mapings)
    cost_m = np.zeros((total,total))
    for i in range(total):
        for j in range(total):
            if i < len(ground_mappings) and j <len(mapings):
                cost_m[i][j] = (distance(ground_mappings[i][0],mapings[j][0],len(ground_mappings)) + distance(ground_mappings[i][1],mapings[j][1],len(ground_mappings)))/2
            else:
                cost_m[i][j] = threshold*2
    
    return cost_m
    