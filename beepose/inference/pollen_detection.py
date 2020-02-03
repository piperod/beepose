import os
import sys 
sys.path.append('..')
from beepose.utils.util import NumpyEncoder, rotate_bound2,distance_point,distance_line_point, read_json,save_json, dets2boxes,boxes2dets,non_max_suppression_slow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import json
import glob,os
import pylab
import cv2
from keras.models import load_model
import argparse
import math

FPS = 20.0

def pollen_samples(mappings,annotations):
    
    ids=list(mappings.keys())
    
    pollen_ids={}
    
    for idx in ids:
        pol = annotations[annotations['#frame']==idx]
        ps=pol[pol['pollen']==1][['tagx','tagy']]
        if  len(ps)>0:
            pollen_ids[idx]=ps.values  
    return pollen_ids

def npollen_samples(mappings,annotations):
    
    ids=list(mappings.keys())
    
    pollen_ids={}
    
    for idx in ids:
        pol = annotations[annotations['#frame']==idx]
        ps=pol[pol['pollen']==0][['tagx','tagy']]
        if  len(ps)>0:
            pollen_ids[idx]=ps.values  
    return pollen_ids

def find_matchings_pollen(mappings,pollen_samples):
    matching_pollen ={}   
    for p in pollen_samples.keys():
        ps=pollen_samples[p]
        mappings_frame = mappings[p]
        pol=ps[0] 
        minimum = 15000
        for mapping in mappings_frame: 
            d = (distance_line_point(mapping, pol) + distance_point(mapping[0],pol)+distance_point(mapping[1],pol))/3
            
            if d <minimum: 
                minimum = d
                matching_pollen[p]=mapping
    return matching_pollen

def pollen_pipeling(mappings, annotations):
    
    pollen_samples=pollen_samples(mappings,annotations)
    
    matchings =find_matchings_pollen(mappings,pollen_samples)
    
    return matchings

def filter_trk(trk,threshold=2):
    counter={}
    for k in range(int(np.array(trk).max())):
        counter[k+1]=0
    for f in range(len(trk)):
        for j in trk[f]: 
            if j>0:
                counter[int(j)]+=1
    filtered=[]
    for k in counter.keys():
        if counter[k]<threshold:
            filtered.append(k)
    return filtered




def entering(event):
    d=event['data']
    ent=[]
    for frame in d:
       
        for e in d[frame]: 
            if e['labels']=='entering':
                ent.append(int(float(e['id'])))
    return ent       




def load_for_pollen(folder,detections_path,track_path,video_path,folder_video ='/mnt/storage/Gurabo/videos/Gurabo/mp4', model_json='../BeeLab/2l_model.json',model_weights='../BeeLab/2l_model.h5'):
    path=detections_path
    detections = read_json(os.path.join(folder,path))
    path2=track_path
    trk= read_json(os.path.join(folder,path2))  # raw tracking 
    video =video_path
    vidcap = cv2.VideoCapture(video)
    model=load_model(model_json,model_weights)
    return path,detections,trk,vidcap,model

def pollen_classifier_fragment(detections_file,trk_file,video_file,model_file,gpu,gpu_fraction,trk_pollen_name,start=0,limit=72000):
    
    """
    Function to process a fragment of a video with pollen detection. The model takes as input 
    
    Inputs : 
        - detections_file : path to detection file. 
        - trk_file : path to tracking file 
        - model_file : path to the model to be used for prediction
        - gpu : Number id of the gpu to use. 
        - gpu_fraction : fraction of the gpu
        
    """
    if type(gpu)==int:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]="%d"%gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
    session = tf.Session(config=config)
    folder = '/'.join(detections_file.split('/')[:-1])
    model = load_model(model_file,compile=False)
    
        
    print(detections_file)
    print(trk_file)
    print(video_file)
    detections = read_json(detections_file)
    
    trk = read_json(trk_file)
    
    print('Staring trk pollen classifier')
    trk_pollen={}
    for trk_frame in trk:
        for t in trk_frame:
            trk_pollen[t]=[]
            
            
            
    vidcap = cv2.VideoCapture(video_file)
    vidcap.set(cv2.CAP_PROP_POS_MSEC,start*1000.0/FPS)
    
    for i,k in enumerate(range(start,limit)):
        if k%500 ==0:
            print(k)
        if k%1000==0 and k>0:
            print('checkpoint at frame:',k)
            save_json(os.path.join(folder,trk_pollen_name),trk_pollen)
        try:
            mappings=detections[str(k)]['mapping']
            parts=detections[str(k)]['parts']['1']
            dets = detections[str(k)]['parts']['2']
            boxes = dets2boxes(dets,size=20)
            dets = boxes2dets(non_max_suppression_slow(boxes,0.6)[::-1])
        except:
            print('Problem with ',k ,str(k))
            continue 
        thrx = [t[:2] for t in dets]
       
        success,image = vidcap.read()
        RGB=image
        for p in parts:
            if p[0]<350 or p[0]> 2200 or p[1]>1200 :continue 
            for m in mappings:
                if m[-1][0]==3 and m[-1][1]==2 and m[1]==p[:2]:
                    thorax = m[0]
                    try:
                        idx=thrx.index(thorax)
                    except:
                        print('thorax not found')
                        continue
                    t_id=trk[int(k)][idx]
                    #print(t_id)
                    for m in mappings:
                        if m[-1][0]==1 and m[-1][1]==2 and m[1]==p[:2]:
                            if m[0][1]<100:continue 
                            rectangle=[250,300]
                            myradians = math.atan2(m[0][1]-m[1][1] ,m[0][0]-m[1][0]) 
                            angle=math.degrees(myradians)-90
                            im=rotate_bound2(RGB,((m[0][0]+m[1][0])/2),((m[0][1]+m[1][1])/2),angle,rectangle[0],rectangle[1])
                            im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
                            score=model.predict(np.array([cv2.resize(im,(180,300))]))

                            if score[0][1]>0.5:
                                #plt.title(str(1)+'_'+str(t_id)+'_'+str(score[0][1])+'_'+str(k))
                                trk_pollen[t_id].append([1,score[0][1],k])
                                #plt.imshow(im)
                                #plt.show()
                                #clear_output(wait=True)
                            else:
                                #plt.title(str(0)+'_'+str(t_id)+'_'+str(score[0][1])+'_'+str(k))
                                trk_pollen[t_id].append([0,score[0][0],k])
                                #plt.imshow(im)
                                #plt.show()
                                #clear_output(wait=True)
                                
    print('checkpoint at frame:',k)
    save_json(os.path.join(folder,trk_pollen_name),trk_pollen)
    
def launch_pollen_folder(folder,folder_video,pathportion,division): 
    
    """
    This function process all the videos in a given folder. 
    """
    
    files = glob.glob(os.path.join(folder,'merged*.json'))
    total = len(files)
    init = int((portion-1)*division*total)
    end = int(portion*division*total)
    print(total,init,end)
    print(files)
    processed = glob.glob(os.path.join(folder,'trk_pollen_raw_*.json'))
    processed_files = [f.split('/')[-1] for f in processed]
    for file in files[init:end]:
        print(file)
        path,detections,trk,vidcap,model = load_for_pollen(folder,folder_video=folder_video)
        pollen_classifier(trk,folder,path,detections,vidcap,model)
                                
                                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default ='detections/one_week/', help='input detections file')
    parser.add_argument('--folder_video',type=str, default= '/mnt/storage/Gurabo/videos/Gurabo/mp4',help='Folder where videos are')
    parser.add_argument('--no_process_again',type=bool, default=True, help = 'If its processed not process again')
    parser.add_argument('--day',type=int, default = 21, help='day to process')
    parser.add_argument('--hour',type= int , default= 8, help='Hour')
    parser.add_argument('--endhour',type= int , default= 18, help='Hour')
    parser.add_argument('--division',type=float, default =0.25,help='divide the  list to process')
    parser.add_argument('--portion',type=float, default =1,help='part of  the  list to process')
    
    args = parser.parse_args()
    folder = args.folder
    folder_video = args.folder_video
    proc = args.no_process_again
    day = args.day
    hour = args.hour
    end = args.endhour
    division = args.division
    portion = args.portion
    
    
    
    
