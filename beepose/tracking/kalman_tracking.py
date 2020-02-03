import os.path
import sys
sys.path.append('..')

import numpy as np
import time
import argparse
import pandas as pd 
import numpy as np
import json
import glob,os
import math
from beepose.tracking.sort import  * 
from scipy.spatial import distance

def nms(det_id,pos, posList,threshold):
    
    """ 
    Non Maxima supression algorithm
    """
        
    for det_id2,pos2 in enumerate(posList):

        if (det_id2 == det_id): 
            continue
        if (det_id2!=det_id):

            dis = distance.euclidean(pos2[0:2], pos[0:2])
            #print(dis)
            if (dis<threshold):

                if (pos2[2]>pos[2]):
                    return False

            #return False
    return True

def convert_detections_json_to_csv(dets_json,w,h,json_dets_file=None,dets_csv = None,use_nms=True,part='2'):



    if (json_dets_file is None):
        json_dets_file = dets_json

    if (dets_csv is None):
        dets_csv= json_dets_file+'.dets.txt'


    with open(dets_json) as f:
        dets_json = json.load(f)

    keylist = list(dets_json.keys())
    out={}
    
    with open(dets_csv,'w') as out_file:
        for i in range(len(keylist)):
            frame = keylist[i]

            out[frame]=[]

            frameitem=dets_json[frame]
            parts=frameitem['parts'][part] # 2==thorax

            for det in parts:
                thorax = (det[0],det[1],det[2])
                out[frame].append(thorax)

        for i in range(len(out)):
            frame = keylist[i]
            for det_id,pos in enumerate(out[frame]):
                if (use_nms):
                    if (nms(det_id,pos,out[frame],9.0) is not True):
                        continue

                    print('%d,-1,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(int(frame),pos[0]-w/2,pos[1]-h/2,w,h),file = out_file)




    


    #         dets.append([int(frame),det_id,pos[0]-w/2,pos[1]-h/2,w,h, 1, -1,-1,-1])
    # seq_dets=np.array(dets)
    # return seq_dets



def do_tracking(dets_json,dets_csv, tracks=None):
    
    """ 
    This is the main function for doing kalman tracking. It takes as input  the converted detections and outputs 
    kalman_tracks and id_kalman_tracks that holds the same structure as trk and id_trk files from hungarian algorithm. 
    
    Inputs: 
    
        -dets_csv : converted csv file 
        - tracks : name of the output
    """

    if (tracks is None):
        tracks = dets_json[:-4]+'kalman_tracks.json'
        tracks_idf = dets_json[:-4]+'id_kalman_tracks.json'

    seq_dets = np.loadtxt(dets_csv,delimiter=',') #load detections
    total_time = 0.0
    total_frames = 0
    trajectory=[]


    mot_tracker = Sort() #create instance of the SORT tracker
    KalmanBoxTracker.count=0
   

   
    kalman_tracks ={}
    tracks_id = {}
    for frame in range(int(seq_dets[:,0].max())+1):
        if frame>10 and frame%10000 ==0: 
            print('Processing frame %02d'%frame)
        if frame not in kalman_tracks.keys():
            kalman_tracks[frame]={}
        dets = seq_dets[seq_dets[:,0]==frame,2:7]
        dets[:,2:4] += dets[:,0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        total_frames += 1

        start_time = time.time()
        trackers_states = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time
            
        for d in trackers_states:
            
            x = d[0]+200
            y = d[1]+200
            key= str(int(x))+'_'+str(int(y))
            kalman_tracks[frame][key]=d[4]
            
            if d[4] not in tracks_id.keys():
                tracks_id[d[4]]={'positions':[[x,y]],
                                  'init_frame':frame,
                                  }
            else:
                tracks_id[d[4]]['positions'].append([x,y])
            
    with open(tracks,'w') as tracks_fid: 
        json.dump(kalman_tracks,tracks_fid)
    with open(tracks_idf,'w') as tracks_f:
        json.dump(tracks_id,tracks_f)
        #print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file = tracks_fid)
    #print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))

def kalman_tracking (dets_json, dets_csv='',box_size =[400,400],part='2'):
    """
    This function takes as input the detections in json format. Transforms the detection to match sort algorithm and 
    performs kalman tracking by using sort algorithm. 
    
    """
    if dets_csv == '':
        dets_csv = dets_json+'.dets.txt'
    print('Reading File :', dets_json)
    print('Saving converted  file in:',dets_csv)
    
    convert_detections_json_to_csv(dets_json,box_size[0],box_size[1],json_dets_file=None,dets_csv = None,use_nms = True,part=part)
    do_tracking(dets_json,dets_csv, tracks= None)

def kalman_tracking_folder (folder,box_size=[400,400]):
    """
    Function to perform kalman tracking on all the detection files that are contained in a folder. 
    
    
    """
    files = glob.glob(os.path.join(folder,'merged*.json'))
    for f in files:
        print(f)
        kalman_tracking(f,box_size=box_size)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dets_json', default ='merged_C02_170621100000_fine_new.json', help='input detections file')
    parser.add_argument('--dets_csv', default='', help='output name file, by default converted detections file name')
    parser.add_argument('--box_size',default=[400,400],help='Size of the box for sort')
    parser.add_argument('--output_tracks', default='', help='output name file, by default track+detections file name')
    parser.add_argument('--folder_detections',default='', help= ' If it is given perfom the kalman tracking on a set of detections files that are in a folder. ')
    
    args = parser.parse_args()
    
    folder = args.folder_detections
    if folder != '': 
        print('Doing the kalman tracking in a folder')
        kalman_tracking_folder(folder,box_size)
    else: 
        dets_json = args.dets_json
        path = args.output_tracks
        dets_csv = args.dets_csv
        output = args.output_tracks
        box_size = args.box_size
        kalman_tracking(dets_json,dets_csv,box_size)
    