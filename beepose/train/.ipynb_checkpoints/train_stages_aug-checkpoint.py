import sys
sys.path.append("..")
import os
import pandas
import re
import math
import argparse
from beepose.models.train_model import get_training_model_new
from beepose.train.ds_iterator import DataIterator
from beepose.train.ds_client_generator import DataGeneratorClient
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard
from keras.layers.convolutional import Conv2D
from keras.applications.vgg19 import VGG19

import numpy as np
from beepose.data.pose_dataset import get_dataflow,get_dataflow_batch
from beepose.data.pose_augmenter import set_network_input_wh, set_network_scale,check_network_input_wh
from beepose.utils.util import save_json
from beepose.train.inference_model import save_inference_model
from beepose.utils.util import get_skeleton_from_json, get_skeleton_mapIdx
#from beepose.train.train_callbacks import TensorBoardImage
import cv2
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152



def get_last_epoch(TRAINING_LOG):
    data = pandas.read_csv(TRAINING_LOG)
    return max(data['epoch'].values)

def get_eucl_loss(batch_size):
    def eucl_loss(x, y):
        return K.sum(K.square(x - y)) / batch_size / 2
    return eucl_loss


            
def gen(df,batch_size,stages):
    batch_data =df.get_data()
    while True:
        batches_x, batches_x1, batches_x2, batches_y1, batches_y2 = \
            [None]*batch_size, [None]*batch_size, [None]*batch_size, \
            [None]*batch_size, [None]*batch_size
        # Img: Batch of images, *_weights:Weights obtained from mask
        # heat_maps : confidence maps for parts
        # Vector_maps: part affinity fields
        img,heat_weights,vec_weights, heat_maps,vector_maps  = tuple(next(df.get_data()))
        coin_toss= np.random.uniform(0,1)
        #if coin_toss <0.1:
        #    dta_img = img[0,:,:,:]
        #    mask_img = heat_weights[0,:,:,:]
        #    plt.imsave('random_img.png',dta_img)
        #    plt.imsave('vec_weights.png',mask_img[:,:,2])
            #heatmap = cv2.resize(np.array(heat_maps[0, :, :, 1],dtype=np.float), (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
            #heat=plt.imshow(dta_img[:,:,:])
            #heat+=plt.imshow(heatmap[:,:], alpha=.5)
            #plt.imsave('heatmap.png',heat)
        batch_x= img
        vec_weights=np.ones(vector_maps.shape)
        heat_weights=np.ones(heat_maps.shape)
        #print('SHAPE',heat_maps.shape)
        array=[vector_maps,heat_maps,
               vector_maps,heat_maps,
               vector_maps,heat_maps,
               vector_maps,heat_maps,
               vector_maps,heat_maps,
               vector_maps,heat_maps]
        #import pdb;pdb.set_trace()
        yield [img,vec_weights,  heat_weights], array[:stages*2]
        
def prepare_generators(ann_path,imgs_path, stages=2, batch_size=10, network_scale=8,w=368,h=368,  mins=0.125,
    maxs=0.6,mina=-np.pi/4,maxa=np.pi/4,sigma=16,translation=True,rotation=True,scale=True):
    # prepare generators
    
    set_network_input_wh(w, h)
    set_network_scale(network_scale)
    output_shape = (h,w)
    print('output_shape',output_shape)
    print('network input ', check_network_input_wh())
    df = get_dataflow(ann_path, True, imgs_path)
    train_samples = df.size()
    
    print('Training on %d samples'%train_samples)
    batch_df = get_dataflow_batch(ann_path,is_train=True,
                        batch_size=batch_size,img_path=imgs_path,
                        output_shape=output_shape,
                     translation=translation,rotation=rotation,
                     scale=scale,mins=mins,maxs=maxs,mina=mina,maxa=maxa,sigma=sigma)
    
    return gen(batch_df,batch_size,stages),train_samples


def prepare_logging(folder,stages,np1,np2,params):
    # Prepare logging
    os.makedirs(folder,exist_ok=True)
    WEIGHTS_100_EPOCH = os.path.join(folder,"weights-{epoch:06d}_%d_%d_%d.h5"%(stages,np1,np2))
    WEIGHTS_BEST = os.path.join(folder,"weights_%d_%d_%d.best.h5"%(stages,np1,np2))
    WEIGHTS_COMPLETE = os.path.join(folder,"complete_model_%d_%d_%d.h5"%(stages,np1,np2))
    TRAINING_LOG = os.path.join(folder,"training_new_%d_%d_%d.csv"%(stages,np1,np2))
    LOGS_DIR = os.path.join(folder,"logs/")
    save_json(os.path.join(folder,'model_params.json'),params)
    os.makedirs(LOGS_DIR,exist_ok=True)
    return WEIGHTS_100_EPOCH,WEIGHTS_BEST,WEIGHTS_COMPLETE,TRAINING_LOG,LOGS_DIR


def vgg_layers():
    vgg_layers = dict()
    vgg_layers['conv1_1'] = 'block1_conv1'
    vgg_layers['conv1_2'] = 'block1_conv2'
    vgg_layers['conv2_1'] = 'block2_conv1'
    vgg_layers['conv2_2'] = 'block2_conv2'
    vgg_layers['conv3_1'] = 'block3_conv1'
    vgg_layers['conv3_2'] = 'block3_conv2'
    vgg_layers['conv3_3'] = 'block3_conv3'
    vgg_layers['conv3_4'] = 'block3_conv4'
    vgg_layers['conv4_1'] = 'block4_conv1'
    vgg_layers['conv4_2'] = 'block4_conv2'
    return vgg_layers



def get_call_backs(lrate,step_decay,val_dir,WEIGHTS_BEST,WEIGHTS_100_EPOCH,WEIGHTS_COMPLETE,TRAINING_LOG,LOGS_DIR):
    lrate = LearningRateScheduler(step_decay)
    checkpoint = ModelCheckpoint(WEIGHTS_BEST, monitor='loss', verbose=0, save_best_only=False, save_weights_only=True, mode='min', period=1)
    checkpoint2 = ModelCheckpoint(WEIGHTS_100_EPOCH, monitor='loss', verbose=0, save_best_only=False, save_weights_only=True, mode='min', period=100)
    checkpoint3 = ModelCheckpoint(WEIGHTS_COMPLETE, monitor='loss', verbose=0, save_best_only=True,save_weights_only=False, mode='min', period=1,)
    csv_logger = CSVLogger(TRAINING_LOG, append=True)
    tb = TensorBoard(log_dir=LOGS_DIR, histogram_freq=0, write_graph=True, write_images=False)
    #tb_image = TensorBoardImage(val_dir,LOGS_DIR)
    callbacks_list = [lrate, checkpoint, csv_logger, tb,checkpoint2,checkpoint3]
    return callbacks_list

def get_step_decay(base_lr, iterations_per_epoch, gamma, stepsize):
    def step_decay(epoch):
        initial_lrate = base_lr
        steps = epoch * iterations_per_epoch
        lrate = initial_lrate * math.pow(gamma, math.floor(steps/stepsize))
        return lrate
    return step_decay


def setup_lr_multipliers(model):
    # setup lr multipliers for conv layers
    lr_mult=dict()
    for layer in model.layers:

        if isinstance(layer, Conv2D):

            # stage = 1
            if re.match("Mconv\d_stage1.*", layer.name):
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 1
                lr_mult[bias_name] = 2

            # stage > 1
            elif re.match("Mconv\d_stage.*", layer.name):
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 4
                lr_mult[bias_name] = 8

            # vgg
            else:
                kernel_name = layer.weights[0].name
                bias_name = layer.weights[1].name
                lr_mult[kernel_name] = 1
                lr_mult[bias_name] = 2
    return lr_mult
    

def main():   
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', type=int, default =6, help='number of stages')
    
    parser.add_argument('--folder',type=str,default="weights_logs/5p_6/",help='"Where to save this training"' )
    parser.add_argument('--gpu',default =1, help= 'what gpu to use, if "all" try to allocate on every gpu'  )
    parser.add_argument('--gpu_fraction', type=float, default =0.9, help= 'how much memory of the gpu to use' )
#     parser.add_argument('--np1', type=int, default =12, help= 'Number of pafs' )
#     parser.add_argument('--np2', type=int, default =6, help= 'number of heatmaps' )
    parser.add_argument('--ann', type=str,default = '../../data/raw/bee/dataset_raw/train_bee_annotations2018.json' ,help =' Path to annotations')
    parser.add_argument('--imgs',type=str, default = '../../data/raw/bee/dataset_raw/train',help='Path to images folder')
    parser.add_argument('--val_imgs',type=str,default='../../data/raw/bee/dataset_raw/validation',help= 'path to val images folder')
    parser.add_argument('--batch_size', type=int, default =10, help= 'batch_size' )
    parse.add_argument('--max_iter', type=int,default=20000, help='Number of epochs to run ')
    
    
    args = parser.parse_args()
    folder = args.folder
    stages=int(args.stages)
    val_imgs = str(args.val_imgs)
    ann_file = str(args.ann)
    
    numparts, skeleton = get_skeleton_from_json(ann_file)
    
    fraction = float(args.gpu_fraction)
    
    np2= numparts + 1#6#number of channels for parts
    np1= np2 * 2#12 #number of channels for pafs
#     numparts = np2 
    mapIdx = get_skeleton_mapIdx(skeleton)
    gpu = int(args.gpu)
    batch_size = int(args.batch_size)
    
    base_lr = 4e-5 
    momentum = 0.9
    weight_decay = 5e-4
    lr_policy =  "step"
    gamma = 0.333
    stepsize = 68053
    max_iter = args.max_iter
    params={'stages':stages,
             'np1':np1,
             'np2':np2, 
             'numparts':numparts,
             'skeleton': skeleton,
             'mapIdx':mapIdx,
              'gpu':gpu,
              'batch_size':batch_size,
              'base_lr':base_lr,
              'momentum':momentum,
              'weight_decay':weight_decay,
              'gamma':gamma,
              'stepsize':68053
               }
    
    print(gpu)
    
#         print(gpu)
#         os.environ["CUDA_VISIBLE_DEVICES"]="%d"%gpu
        
    
    
    
    config = tf.ConfigProto()
    if gpu != 'all':
        config.gpu_options.visible_device_list= "%d"%gpu
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]="%d"%gpu
    config.gpu_options.per_process_gpu_memory_fraction = fraction
    session = tf.Session(config=config)
    
    # Prepare generator
  
    train_gen,train_samples = prepare_generators(ann_path=str(args.ann),imgs_path=str(args.imgs), stages=stages, batch_size=batch_size)
    
    WEIGHTS_100_EPOCH,WEIGHTS_BEST,WEIGHTS_COMPLETE,TRAINING_LOG,LOGS_DIR =prepare_logging(folder,stages,np1,np2,params)
    
    

    model = get_training_model_new(weight_decay,np1=np1,np2=np2,stages=stages)

    layers_vgg = vgg_layers()

    # load previous weights or vgg19 if this is the first run
    if os.path.exists(WEIGHTS_BEST):
        print("Loading the best weights...")

        model.load_weights(WEIGHTS_BEST)
        last_epoch = get_last_epoch(TRAINING_LOG) + 1
    else:
        print("Loading vgg19 weights...")

        vgg_model = VGG19(include_top=False, weights='imagenet')

        for layer in model.layers:
            if layer.name in layers_vgg:
                vgg_layer_name = layers_vgg[layer.name]
                layer.set_weights(vgg_model.get_layer(vgg_layer_name).get_weights())
                print("Loaded VGG19 layer: " + vgg_layer_name)

        last_epoch = 0

    


    lr_mult = setup_lr_multipliers(model)

    # configure loss functions 

    losses = {}
    for i in range(1,stages+1):
        losses["weight_stage"+str(i)+"_L1"] = get_eucl_loss(batch_size)
        losses["weight_stage"+str(i)+"_L2"] = get_eucl_loss(batch_size)
    print(losses.keys())
   
    # learning rate schedule - equivalent of caffe lr_policy =  "step"
    iterations_per_epoch = train_samples // batch_size
    
    step_decay = get_step_decay(base_lr, iterations_per_epoch, gamma, stepsize)
    

    # configure callbacks
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = get_call_backs(lrate,step_decay,val_imgs,
                                 WEIGHTS_BEST,WEIGHTS_100_EPOCH,
                                    WEIGHTS_COMPLETE,TRAINING_LOG,LOGS_DIR)
    # sgd optimizer with lr multipliers
    #multisgd = MultiSGD(lr=base_lr, momentum=momentum, decay=0.0, nesterov=False, lr_mult=lr_mult)
    multisgd = Adam(lr=base_lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # start training
    model.compile(loss=losses, optimizer=multisgd, metrics=["accuracy"])

    model.fit_generator(train_gen,
                    steps_per_epoch=train_samples// batch_size,
                    epochs=max_iter,
                    callbacks=callbacks_list,
                    #validation_data=val_di,
                    #validation_steps=val_samples // batch_size,
                    use_multiprocessing=False,
                    initial_epoch=last_epoch
                    )
    
    ### Putting the model in testing mode.
    OUTPUT_PATH = os.path.join(folder,'Inference_model.h5')
    save_inference_model(stages,np1,np2,WEIGHTS_BEST,OUTPUT_PATH)
    
if __name__ == '__main__':
    main()
