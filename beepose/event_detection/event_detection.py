import glob,os,sys
sys.path.append('..')
import json
import cv2
import numpy as np
import pylab
import pandas as pd 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from beepose.utils.util import *
import re
import argparse


def id_frame(trk):
    trk_frame={}
    for ids in trk.keys():
        trk_frame[trk[ids]['init_frame']]=[]
        for i in range(trk[ids]['init_frame']+1,trk[ids]['init_frame']+len(trk[ids]['positions'])):
            trk_frame[i]=[]
    for ids in trk.keys():
        trk_frame[trk[ids]['init_frame']].append({'id':ids,'positions':trk[ids]['positions']})
        for i in range(trk[ids]['init_frame']+1,trk[ids]['init_frame']+len(trk[ids]['positions'])):
            trk_frame[i].append({'id':ids,'positions':trk[ids]['positions']})
            
    return trk_frame

def getimg(frame):
    video.set(cv2.CAP_PROP_POS_MSEC,frame*1000.0/FPS)
    _,img=video.read(); img=img[...,[2,1,0]]
    return img


def show_track(track,ids):
    frame = track[ids]['init_frame']
    pts = np.array(trk[ids]['positions'])
    plt.imshow(getimg(frame))
    for j in range(0,len(pts)-1,2):
        plt.plot((pts[j][0],pts[j+1][0]),(pts[j][1],pts[j+1][1]),'-.',color='white')
    for j in range(1,len(pts)-2,2):
        plt.plot((pts[j][0],pts[j+1][0]),(pts[j][1],pts[j+1][1]),'-.',color='white')
    plt.plot(pts[:,0],pts[:,1],'.r')
    plt.xticks(np.arange(0, 2600, 50.0))
    plt.yticks(np.arange(0, 1440, 50.0))
    plt.grid()

def show_track_frame(trk_frame,frame):
    frames = trk_frame[frame]
    plt.imshow(getimg(frame))
    for f in frames:
        pts = np.array(f['positions'])
        for j in range(0,len(pts)-2,2):
            plt.plot((pts[j][0],pts[j+1][0]),(pts[j][1],pts[j+1][1]),'--',color='white')
            plt.plot((pts[j+1][0],pts[j+2][0]),(pts[j+1][1],pts[j+2][1]),'--',color='white')
        plt.plot(pts[:,0],pts[:,1],'o')
    plt.xticks(np.arange(0, 2600, 50.0))
    plt.yticks(np.arange(0, 1440, 50.0))
    plt.grid()
    
def showf(frame,part=2):
    pts=np.array(tracks[str(frame)]['parts'][str(part)])
    plt.imshow(getimg(frame))
    plt.plot(pts[:,0],pts[:,1],'.r')
    plt.xticks(np.arange(0, 2560, 50.0))
    plt.yticks(np.arange(0, 1440, 50.0))
    plt.grid()
    print(pts[:,0],pts[:,1])
    
    
def track_classification(trk, inside=200,outside=1050,POS=5):
    """
    Function to perform track classification. It is based on the start, end and lenght of the trajectory. 
    """
    n, i, i_o, i_r, r_r, r_o, r_i , o, o_i, o_r ,i_r_i, o_r_o  = [],[],[],[],[],[],[],[],[],[],[],[]
    for key in trk.keys():
        positions = trk[key]['positions']
        positions_array = np.array(positions)[:,1]
        start = positions[0][1]
        end = positions[-1][1]
        if len(positions)>POS:
            # Inside 
            if start < inside and end < inside :
                i.append(key)
            # Outside 
            elif start > outside and end > outside:
                o.append(key)
            # Inside-outside 
            elif start < inside and end > outside: 
                i_o.append(key)
            # Ramp Ramp
            elif start > inside and start < outside and end >inside and end < outside :
                r_r.append(key)
            
            # Ramp - Outside
            elif start> inside  and start < outside and end > outside: 
                r_o.append(key)
            # Ramp - Inside
            elif start < outside and start > inside and end <inside: 
                r_i.append(key)
            # Outside Inside 
            elif start > outside and end < inside: 
                o_i.append(key)
            # Outside Ramp
            elif start > outside  and end > inside and end < outside:
                o_r.append(key)
            #Inside Ramp
            elif start < inside and end >inside and end < outside: 
                i_r.append(key)
            # Inside Ramp Inside
            elif start < inside and end < inside and np.any(positions_array > inside+100) :
                i_r_i.append(key)
            # Outside Ramp Outside
            elif start > outside and end > outside and np.any(positions_array > outside-200):
                o_r_o.append(key)
        else: 
            #noise 
            n.append(key)
    return  n, i, i_o, i_r, r_r, r_o, r_i , o, o_i, o_r ,i_r_i, o_r_o 

def load_trk(root):
    """
    Loads the trk file according to the format used. 
    v1: The information is indexed by id of the track. 
    v2. The information of the id matching v1 is inside the key "data". The information of the raw tracking is under "frames"
    
    """
    if 'v2' in root: 
        with open(root) as file:
            trk= json.load(file)
        return trk['data']
    else:
        with open(root) as file:
            trk= json.load(file)
        return trk 
        
def filter_pollen_ids(pollen_ids,trk):
    filter_ids=[]
    for p in pollen_ids:
        if p not in trk.keys(): 
            p = str(int(float(p)))
            if p not in trk.keys():
                print(p)
                continue
            
        positionsy = np.array(trk[p]['positions'])[:,1]
        positionsx = np.array(trk[p]['positions'])[:,0]
        desp_max_x = max(positionsx)-min(positionsx)
        desp_max_y = max(positionsy)-min(positionsy)
        if desp_max_y < 200:
            continue 
        if max(positionsx)>2200 and min(positionsx)>2000:
            continue
        if min(positionsy)<850:
            filter_ids.append(p)
    return filter_ids 
        
    
def clean_pollen_detections(filepath):
    pollen_trk = read_json(filepath)
    pollen_ids = []
    for k in pollen_trk.keys():
        if pollen_trk[k]==[]:continue
        indicative = np.array(pollen_trk[k])[:,0].mean()
        if indicative > 0.5 and len(pollen_trk[k])>3:
            pollen_ids.append(k)
    return  pollen_ids,pollen_trk 
        
def events_update(event_file,id_trk_file,count_file,trk_class_file):
    pollen_ids,pollen_trk = clean_pollen_detections(event_file)
    trk= read_json(id_trk_file)
    pollen_ids = filter_pollen_ids(pollen_ids,trk)
    count = read_json(count_file)
    trk_class =read_json(trk_class_file)
    count['pollen'] = len(pollen_ids)
    save_json(count_file,count)
    trk_class['pollen']=pollen_ids
    save_json(trk_class_file,trk_class)

    
def event_detection(root,limit=7200,start=0,frames=72000,inside=200,outside=1050,save=True,output_folder=''):
    
    """
    This function save the events on a frame by frame fashion. You can specify the limit of the event frame. As well as the start. 
    
    inputs : 
    
        root : Tracking file path 
        limit : How many frames  to process
        start : where (which frame)  to start the event detection 
        frames : the length of the videos 
        inside: Defintion of inside or the upper limit 
        outside : Definition of the outside of the bottom limit
        save : whether to save the results
        output_folder : Where to save the events. 
        
    Outputs: 
    
        Event_v2 : The event format compatible with the interface
        Trk_class : File containing what class was given to each of the tracks ids. 
        Count_v2 : The total aggregated count of events per hour.  
        
        
    """
    print('loading trk')
    trk=load_trk(root)
    end=start+limit
    array= {}
    n, i, i_o, i_r, r_r, r_o, r_i , o, o_i, o_r ,i_r_i, o_r_o = track_classification(trk, inside = inside, outside = outside, POS = 5)
    #for j in range(start,start+limit+1):
    #    array[j]=[]
    print('track classification')
    track_class = {}
    track_class['foraging']={}
    track_class['track_shape']={}
    track_class['foraging']['leaving'] = i_o + r_o
    track_class['foraging']['entering'] = o_i+o_r
    track_class['foraging']['entering-leaving'] = o_r_o
    track_class['track_shape']['inside'] = i
    track_class['track_shape']['inside_out'] = i_o
    track_class['track_shape']['inside_ramp'] = i_r
    track_class['track_shape']['ramp_ramp'] = r_r
    track_class['track_shape']['ramp_outside'] = r_o
    track_class['track_shape']['ramp_inside'] = r_i
    track_class['track_shape']['outside'] = o
    track_class['track_shape']['outside_inside'] = o_i
    track_class['track_shape']['outside_ramp'] = o_r
    track_class['foraging']['walking'] = i_r_i
    track_class['track_shape']['noise']=n
    ID=0
    for l,tr in enumerate(trk):
        t=trk[tr]
        init=t['init_frame']
        if init< start:continue
        if init>end:continue
        finish=init+len(t['positions'])
        foraging = track_class['foraging']
        pos=t['positions']
        frame_detection=[]
        ID+=1
        for j in range(len(pos)):
            
            evts={}
            evts['time']=float(init+j)/20
            evts['id']=str(tr)
            evts['frame']=str(init+j)
            evts['x']=pos[j][0]-344.781/2
            evts['y']=pos[j][1]-344.781/2
            evts['cx']=pos[j][0]
            evts['cy']=pos[j][1]
            evts['width']=344.781
            evts['height']=344.781
            evts['angle']=90
            evts['bool_acts']=['false','false','false','false']
            evts['notes']=''
            if tr in foraging['leaving'] :
                event=['leaving']
            elif tr in foraging['entering']: 
                event=['entering']
            elif tr in foraging['entering-leaving']:
                event= ['entering-leaving']
            elif tr in foraging['walking']:
                event= ['walking']
            else: 
                event= ['Not_Event']
                
            evts['labels']=','.join(event)
            evts['classifier']={'entering_leaving':{
                                    event[0]:{'length':len(pos),
                                             'start_position':pos[0],
                                             'end_position':pos[-1]}}}
            if int(init+j) not in array.keys():
                array[int(init+j)]=[]
            array[int(init+j)].append(evts)
            
    obj = {
        'info':{'type':'events-multiframe'},
        'data':array
    } 
    print('counting')
    counts={}
    for key in track_class.keys():
        for k in track_class[key].keys():
            counts[k] = len(track_class[key][k])
            
    
    #if start==0 and start+limit >71000:
    name=os.path.join(output_folder,'EVENT_Complete_v2_'+root.split('/')[-1])
    count_name=os.path.join(output_folder,'Count_v2_'+root.split('/')[-1])
    trk_name =os.path.join(output_folder, 'TRK_Class_'+root.split('/')[-1])
        
        
    if save: 
        print('saving')
        with open(name, 'w') as outfile:
            json.dump(obj, outfile)
        with open(count_name,'w') as outfile: 
            json.dump(counts,outfile)
        with open(trk_name,'w') as outfile:
            json.dump(track_class,outfile)
        print('All saved')
    return obj,counts


def do_event_detection_folder(prefix, output_folder, input_folder,limit):
    """
    Event detection applied to a set of files that are in a folder. 
    
    Inputs: 
    
        - prefix : What is the identifier of the Track id file. 
        - output_folder : Where to save the events
        - input_folder : where to find the files according to the prediction
        - limit : In case only a fraction of the file wanted to be processed. 
    """
    
    if output_folder == '':
        output_folder = input_folder #os.path.join(input_folder,'output_events')
    os.makedirs(output_folder,exist_ok=True)
    files  = glob.glob(input_folder+'/'+'*'+prefix+'*')
    processed_all = glob.glob(output_folder+'/Count*')
    
    ##TODO Improve way of checking processed files. Right now is just a hack of the position of the date when parsing the name. But it can change.
    processed_dates = [f.split('/')[-1].split('_')[4] for f in processed_all]#9
    print('files',files)
    print('processed dates',processed_dates)
    for f in files:
        print(f)
        print(f.split('/')[-1].split('_')[2])
        if f.split('/')[-1].split('_')[2] in processed_dates:
            print('File already Processed skipping')
        root=f
        try:
            
            obj,counts=event_detection(root,limit=limit,save=True,output_folder=output_folder)
        except:
            print('Not processed file:',root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default ='id_nms_track_merged_', help='prefix of  ID_TRACK file')
    parser.add_argument('--input_folder',type=str, default = 'detections/one_week',help= 'folder where all the id tracks files are')
    parser.add_argument('--output_folder', type=str, default='', help='output name file, by default input_folder/output_events) file name')
    parser.add_argument('--limit', type = int, default =72000,help = 'Number of frames to process' )
    
    
    args = parser.parse_args()
    
    prefix = args.prefix
    input_folder = args.input_folder
     
    output_folder = args.output_folder
    limit = args.limit
    
    do_event_detection_folder(prefix, output_folder, input_folder,limit)
    
    