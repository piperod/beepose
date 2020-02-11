
import argparse
import cv2
import math
import time
import numpy as np

from scipy.ndimage.filters import gaussian_filter
from beepose.models.train_model import get_testing_model_new
import glob,os

import tensorflow as tf
from beepose.utils.util import NumpyEncoder,save_json,merge_solutions_by_batch
from beepose.inference.inference import inference
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config=config)



    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default ='../../data/raw/bee/dataset_raw/validation.json', help='input images folder')
    parser.add_argument('--output', type=str, default='testing_images_result.json', help='csv file name')
    parser.add_argument('--model', type=str, default='training/weights.best.3p.h5', help='path to the weights file')
    parser.add_argument('--stages',type = int, default = 2, help='How many stages')
    parser.add_argument('--output_folder',type=str,default='detections/validation2019/2p')
    parser.add_argument('--np1',type=int,default=12)
    parser.add_argument('--np2',type=int,default=6)
    parser.add_argument('--resize_factor',type=int,default=4)
    
    

    args = parser.parse_args()
    path = args.folder
    output = args.output
    model_name=args.model 
    keras_weights_file = args.model
    output_folder =args.output_folder
    os.makedirs(output_folder,exist_ok=True)
    
    imlist = sorted([os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')])
    np1=args.np1
    np2 =args.np2
    stages = args.stages
    resize_factor = args.resize_factor 
    tic_total = time.time()
    print('start processing...')
    out= model_name.split('/')[-1].split('-')[-1][:-3]+'.json'
    out_fold = os.path.join(output_folder,str(stages))
    os.makedirs(out_fold,exist_ok=True)
    out_file = os.path.join(out_fold,out)
    print('saving file in ',out_file)
    # load model

    # authors of original model don't use
    # vgg normalization (subtracting mean) on input images
    model = get_testing_model_new(np1=np1,np2=np2,stages=stages)
    model.load_weights(keras_weights_file)
    #model = load_model(model_file_day,compile=False)
    print('model loaded....')
    # load config
    params = { 'scale_search':[1], 'thre1': 
              {0:0.4,1:0.25,
               2:0.4,3:0.4,
               4:0.4,5:0.4,
               5:0.09,6:0.09,
               7:0.01}, 
              'thre2': 0.05, 'thre3': 0.5, 
              'min_num': 4, 'mid_num': 10, 
              'crop_ratio': 2.5, 'bbox_ratio': 0.25} 

    model_params = {'boxsize': 368, 'padValue': 128, 'np': '12', 'stride': 8}  
    frame_detections={} 
    for input_image in imlist:
        tic = time.time()
        print('processing image', input_image)
        # generate image with body parts
        img = cv2.imread(input_image)
        
        img_resized=cv2.resize(img,(img.shape[1]//resize_factor,img.shape[0]//resize_factor))
        
        canvas,mappings,parts = inference(img_resized,model, params, model_params,np1=np2,np2=np1,resize=resize_factor,distance_tolerance=310,numparts=5,
                                                mapIdx=[[0,1],[2,3],[4,5],[6,7]],
                                                limbSeq=[[1,3],[3,2],[2,4],[2,5]])#resize=(256,144))#
        print(parts)
        
        frame_detections[input_image]={}
        #frame_detections[input_image]['detections']=detections
        frame_detections[input_image]['mapping']=mappings
        frame_detections[input_image]['parts']=parts
        print('writing image', input_image)
        toc = time.time()
        print ('Frame processing time is %.5f' % (toc - tic))

    toc_total = time.time()
    print ('Total processing time is %.5f' % (toc_total - tic_total))
    
    import json
    with open(out_file, 'w') as outfile:
        json.dump(frame_detections, outfile,cls=NumpyEncoder)
