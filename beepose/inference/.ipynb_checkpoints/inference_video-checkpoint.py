import sys 
sys.path.append('..')
from keras.models import load_model
from beepose.inference.inference import inference,inference_by_batch
import tensorflow as tf
import argparse
import cv2
import numpy as np
import glob,os,math,time
import json 
from beepose.utils.util import NumpyEncoder,save_json,merge_solutions_by_batch
from beepose.utils import util 
from tqdm import tqdm
from dask import delayed,compute
import logging
import time

logger = logging.getLogger(__name__)

    
def process_video_fragment(file,model_file_day,model_file_nigth,gpu,gpu_fraction,start_frame,end_frame,limbSeq,mapIdx,np1,np2,output,numparts=5,checkpoint=9000,FPS=20):
    """ 
    
    Wrapper function to process video fragment. 
      
      
        
    Inputs: 
    
        - Video: A video capture. 
        - model : a loaded keras model
        - gpu_fraction : Fraction of the GPU to use 
        - start_frame : Frame to start the processing of the video capture
        - end_frame : Frame to end the processing of the video capture 
        - limbSeq: Configuration of the limbs for pafs
        - mapIdx: Configuration of the skeleton 
        - np1: Number of channels on paf branch
        - np2 : Number of channels on heatmaps branch
        
    
    """
    print('arguments:')
   
    if type(gpu)==int:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]="%d"%gpu
        
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
    session = tf.Session(config=config)
    
    model = load_model(model_file_day,compile=False)
    # Adding different threshold for different detection task. For pollen (5,6: 0.09). For tag (0.01)
    params = { 'scale_search':[1], 'thre1': 
              {0:0.4,1:0.25,
               2:0.4,3:0.4,
               4:0.4,5:0.4,
               5:0.09,6:0.09,
               7:0.01}, 
              'thre2': 0.05, 'thre3': 0.4, 
              'min_num': 4, 'mid_num': 10, 
              'crop_ratio': 2.5, 'bbox_ratio': 0.25} 

    model_params = {'boxsize': 368, 'padValue': 128, 'np': '12', 'stride': 8}  
    print('params:',params,
          'model_params',model_params,
          'np1',np1,
          'np2',np2,
          'resize',4,
          'numparts',numparts,
          'mapIdx',mapIdx,
          'limbSeq',limbSeq)
    final_tracks={}
    frame_detections={} 
    resize_factor=4
    frame=0
    tic_total = time.time()
    
    print('Start detections ... ')
    
    show=False
    video = cv2.VideoCapture(file)
    video.set(cv2.CAP_PROP_POS_MSEC,start_frame*1000.0/FPS)
    print('fragment', start_frame,end_frame)
    for idx in tqdm(range(start_frame,end_frame),ascii=True,desc='[%d,%d]--%s'%(start_frame,end_frame,file.split('/')[-1])):
        #print(idx)
        t,im = video.read()
        if t==False:
            print('Error reading frame ',idx)
            continue
        tic = time.time()
        # generate image with body parts
        #params, model_params = config_reader()
        input_image =cv2.resize(im,(im.shape[1]//resize_factor,im.shape[0]//resize_factor))
        frame+=1
        canvas,mappings,parts = inference(input_image, model,params,model_params,show=show,np1=np2,np2=np1,
                                                    resize=resize_factor,limbSeq=limbSeq,mapIdx=mapIdx,numparts=numparts)
        #print(mappings)
        frame_detections[idx]={}
        frame_detections[idx]['mapping']=mappings
        frame_detections[idx]['parts']=parts
        
        toc = time.time()
        logger.debug('Frame processing time is %.5f' % (toc - tic))
        #print ('Frame processing time is %.5f' % (toc - tic))
        if frame%100==0:
            print('writing image', frame)
            #print ('Frame processing time is %.5f' % (toc - tic))
            print('checkpoint saving')
            with open(output, 'w') as outfile:
                json.dump(frame_detections, outfile,cls=NumpyEncoder)
            #with open('track'+output, 'w') as outfile:
            #    json.dump(final_tracks, outfile,cls=NumpyEncoder)
        if frame%int(checkpoint)==0 and frame >0:
            try:
                print('Saving in different file')
                with open(output[:-4]+'_str(frame)_'+output[:-4], 'w') as outfile:
                    json.dump(frame_detections, outfile,cls=NumpyEncoder)
            except:
                print('Problem saving checkpoint')
            #with open('track'+str(frame)+output, 'w') as outfile:
            #    json.dump(final_tracks, outfile,cls=NumpyEncoder)

    toc_total = time.time()
    print ('Total processing time is %.5f' % (toc_total - tic_total))
    print('Final Saving')
    with open(output, 'w') as outfile:
        json.dump(frame_detections, outfile)

        


        
def create_batch(video,frame,batch,params,model_params,resize_factor=4,FPS=20):
    images=[]
    ori =[]
    video.set(cv2.CAP_PROP_POS_MSEC,frame*1000.0/FPS)
    for i in range(batch):
        t,im = video.read()
        if not t:
            print('problem extracting image', i)
            continue 
        oriImg=cv2.resize(im,(im.shape[1]//resize_factor,im.shape[0]//resize_factor))
        ori.append(oriImg)
        multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]
        scale =1
        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                          model_params['padValue'])
        input_img = np.float32(imageToTest_padded)
        images.append(input_img)
    input_imgs = np.array(images)
    return input_imgs,ori


def process_video_by_batch(video, model_file_day,model_file_nigth,gpu,init_frame,size_full_batch,chunk,output_name,limbSeq,mapIdx,resize_factor=4,np1=6,np2=12):
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="%d"%gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    params = { 'scale_search':[1], 'thre1': 0.4, 'thre2': 0.05, 'thre3': 0.4, 'min_num': 4, 'mid_num': 10, 'crop_ratio': 2.5, 'bbox_ratio': 0.25} 
    model_params = {'boxsize': 368, 'padValue': 128, 'np': '12', 'stride': 8} 
    model = load_model(model_file_day,compile=False)
    batch=size_full_batch
    tic_total = time.time()
    batches =[]
    for i in tqdm(range(int(init_frame), int(batch),int(chunk))):
        input_imgs,ori = create_batch(video,i,chunk,params,model_params,resize_factor=resize_factor) 
        output_blobs = model.predict(input_imgs)
        seq0,seq1,seq2,seq3=input_imgs,ori,output_blobs[1],output_blobs[0] 
        result_batch = delayed(inference_by_batch)(seq0,seq1,seq2,seq3,i,params,model_params)
        batches.append(result_batch)
    print('Computing inference')
    results = compute(*batches)
    print('saving')
    final = merge_solutions_by_batch(results)
    save_json(output_name,final)
    toc_total = time.time()
    print ('Total processing time is %.5f' % (toc_total - tic_total))
    return


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default ='/mnt/storage/Gurabo/videos/Gurabo/mp4/C02_170610090000.mp4', help='input video file')
    parser.add_argument('--output', type=str, default='C02_170610090000_test.json', help='csv file name')
    parser.add_argument('--model', type=str, default='training/weights_logs/5p_2/complete_5p_2.best.h5', 
                                help='path to the complete model file model+weights')
    parser.add_argument('--model_config',  default='../../models/pose/complete_5p_2_model_params.json', type=str, help="Model config json file. This file is created when a training session is started. It should be located in the folder containing the weights.")
    parser.add_argument('--GPU',type = int, default =0,choices = [0,1,2])
    #parser.add_argument('--stages', type=int, default= 6, help='number of stages')
    parser.add_argument('--start', type=int, default=0, help='starting frame')
    parser.add_argument('--end', type=int, default=72000, help='end frame')
    parser.add_argument('--gpu_fraction', type=float, default=0.3, help='fraction of the gpu to use')
    #parser.add_argument('--limb_conf',default=[[1,3],[3,2],[2,4],[2,5],[1,2]] )
    #parser.add_argument('--paf_conf',default=[[0,1],[2,3],[4,5],[6,7],[8,9]] )
   
    #________ARGS_________
    
    args = parser.parse_args()
    path = args.video
    print('path',path)
    output = args.output
    model_file = args.model
    start_frame = args.start
    end_frame = args.end
    gpu_fraction = args.gpu_fraction
    #limbSeq = args.limb_conf
    #mapIdx = args.paf_conf
    video = cv2.VideoCapture(path)
    gpu = args.GPU
    
    config_file = args.model_config
    with open(config_file, 'r') as json_file:
        config = json.load(json_file)
    print(config)
    
    limbSeq = config["skeleton"]
    mapIdx = config["mapIdx"]
    sufix = args.sufix
    tracking = args.tracking
    np1 = config["np1"]
    np2 = config["np2"]
    numparts= config["numparts"]
    
    
   
    print('model loaded....')
    # load config
    process_video_fragment(path,model_file,gpu,gpu_fraction,start_frame,end_frame,limbSeq,mapIdx,np1,np2,output)
   
    
