
import glob,os,sys
sys.path.append('..')
from beepose.utils.util import *
import json
from scipy.optimize import linear_sum_assignment as hungarian
import numpy as np
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import argparse



def hungarian_tracking(detections_path,cost,output='',part=str(2)):
    """
    This function executes the hungarian algorithm. It is expecting to receive an instance of detections. Please check documentation for data structure. It will also consider the maximum distance and it outputs in a new file with the data structure explained in documentation. 
    
    Inputs: 
        - detections_path: Path to find detections file
        - cost : Maximum distance allowed
        - output: Optional, if '', will use the same path as were the detections are. 
        - part : Over what part perform the tracking. By default thorax or '2'
    
    """
    if output =='':
        output=detections_path
    detections= read_json(detections_path)
    keylist=list(detections.keys() )
    tracks={}
    final_tracks={}
    cmaps={}
    tracks =np.zeros((len(keylist),70))
    key = keylist[0]
    parts= detections[key]['parts'][part]
    boxes = dets2boxes(parts,size=20)
    parts = boxes2dets(non_max_suppression_slow(boxes,0.6)[::-1])
    track_id={}
    for j in range(len(parts)):
        tracks[0][j]=j+1
        track_id[j+1]={}
        track_id[j+1]['init_frame']=0
        track_id[j+1]['cost']=[0]
        track_id[j+1]['positions']=[parts[j]]
    for i in range(len(keylist)-1):
        key = keylist[i]
        key_next = keylist[i+1]
        parts= detections[key]['parts'][part]
        boxes = dets2boxes(parts,size=20)
        parts = boxes2dets(non_max_suppression_slow(boxes,0.6)[::-1])
        parts_next = detections[key_next]['parts'][part] 
        boxes = dets2boxes(parts_next,size=20)
        parts_next = boxes2dets(non_max_suppression_slow(boxes,0.6)[::-1])
        cmap=cost_matrix_tracks(parts,parts_next,cost)
        cmaps[key]=cmap
        _,idx=hungarian(cmap)
        for j in range(len(parts)): 
            if cmap[j,idx[j]]<cost:
                
                if tracks[i][j]==0:
                    #create new track for detection j at frame i
                    tracks[i][j]=len(track_id.keys())+1
                    
                    k = int(tracks[i][j])# id of the track 
                    track_id[k]={
                        'init_frame':i,
                        'cost':[0],
                        'positions':[parts[j]]
                    }
                    
                tracks[i+1][idx[j]]=tracks[i][j]
                track_id[int(tracks[i+1][idx[j]])]['cost'].append(cmap[j,idx[j]])
                track_id[int(tracks[i+1][idx[j]])]['positions'].append(parts_next[idx[j]])
            
    for k in track_id.keys():
        track_id[k]['mean']=np.array(track_id[k]['cost']).mean()
    
    new_tracks={}
    new_tracks['frames']={}
    new_tracks['data']={}
    for f in range(len(keylist)):
        det=detections[keylist[f]]['parts'][part]
        mapping=detections[keylist[f]]['mapping']
        new_tracks['frames'][f]=[]
        #for j in range(max(tracks[i])):
        for i in range(len(det)):
            
            if int(tracks[f][i])==0:
                continue
            
            else:
                angle=find_angle(det[i],mapping)
                new_tracks['frames'][f].append(tracks[f][i])
                new_tracks['data'][tracks[f][i]] = {'frame':f,
                               'id':tracks[f][i],
                               'location':det[i],
                                'init_frame':track_id[int(tracks[f][i])]['init_frame'],
                                'positions':track_id[int(tracks[f][i])]['positions'],
                                'angle':angle}
    print('saving  trackings') 
    folder = '/'.join(output.split('/')[:-1])
    output1 = os.path.join(folder,'track_nms_v2_'+output.split('/')[-1])
    with open(output1, 'w') as outfile:
        json.dump(new_tracks, outfile,cls=NumpyEncoder)
    output2 = os.path.join(folder,'id_nms_track_'+output.split('/')[-1])
    with open(output2, 'w') as outfile:
        json.dump(track_id, outfile,cls=NumpyEncoder)
        
    
    output3 = os.path.join(folder,'track_nms_'+output.split('/')[-1])
    with open(output3, 'w') as outfile:
        json.dump(tracks, outfile,cls=NumpyEncoder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detections', type=str, default ='C02_170622120000.json', help='input detections file')
    parser.add_argument('--output', type=str, default='', help='output name file, by default track+detections file name')
    parser.add_argument('--cost', type=int, default=200, help='Treshold for tracking')

    args = parser.parse_args()
    path = args.detections
    #limit = args.limit
    output = args.output
    
    cost =args.cost
    
    
    hungarian_tracking(path,cost,output)
        
    
    

    