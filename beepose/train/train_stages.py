import sys
sys.path.append("..")
import os
import pandas
import re
import math
import argparse
from models.train_model import get_training_model_new
from train.ds_iterator import DataIterator
from train.ds_client_generator import DataGeneratorClient
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard
from keras.layers.convolutional import Conv2D
from keras.applications.vgg19 import VGG19




def get_last_epoch():
    data = pandas.read_csv(TRAINING_LOG)
    return max(data['epoch'].values)
# euclidean loss as implemented in caffe https://github.com/BVLC/caffe/blob/master/src/caffe/layers/euclidean_loss_layer.cpp
def eucl_loss(x, y):
    return K.sum(K.square(x - y)) / batch_size / 2

def step_decay(epoch):
    initial_lrate = base_lr
    steps = epoch * iterations_per_epoch
    lrate = initial_lrate * math.pow(gamma, math.floor(steps/stepsize))
    return lrate

if __name__ == '__main__':
    batch_size = 60
    base_lr = 4e-5 # 2e-5
    momentum = 0.9
    weight_decay = 5e-4
    lr_policy =  "step"
    gamma = 0.333
    stepsize = 68053#136106 #// after each stepsize iterations update learning rate: lr=lr*gamma
    max_iter = 20000 # 600000


    #True = start data generator client, False = use augmented dataset file (deprecated)
    use_client_gen = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', type=int, default =6, help='number of stages')
    parser.add_argument('--port', type=int, default =5555, help= 'port where training data is running' )
    parser.add_argument('--folder',type=str,default="weights_logs/5p_6/",help='"Where to save this training"' )
    parser.add_argument('--gpu',default =1, help= 'what gpu to use, if "all" try to allocate on every gpu'  )
    parser.add_argument('--gpu_fraction', type=float, default =0.6, help= 'how much memory of the gpu to use' )
    parser.add_argument('--np1', type=int, default =12, help= 'Number of pafs' )
    parser.add_argument('--np2', type=int, default =6, help= 'number of heatmaps' )
    
    args = parser.parse_args()
    folder = args.folder
    stages=int(args.stages)
    port=int(args.port)
    fraction = float(args.gpu_fraction)
    np1=int(args.np1)#12 #number of channels for pafs
    np2=int(args.np2)#6#number of channels for parts 
    gpu = int(args.gpu)
    print(gpu)
    #stages=2#number of stages of network
    if gpu != 'all':
        print(gpu)
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]="%d"%gpu
    import keras.backend as K
    import tensorflow as tf   
    os.makedirs(folder,exist_ok=True)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = fraction
    session = tf.Session(config=config)
    

    
    WEIGHTS_100_EPOCH = os.path.join(folder,"weights-2-{epoch:04d}_%d_%d_%d.h5"%(stages,np1,np2))
    WEIGHTS_BEST = os.path.join(folder,"weights_%d_%d_%d.best.h5"%(stages,np1,np2))
    WEIGHTS_COMPLETE = os.path.join(folder,"complete_model_%d_%d_%d.h5"%(stages,np1,np2))
    TRAINING_LOG = os.path.join(folder,"training_new_%d_%d_%d.csv"%(stages,np1,np2))
    LOGS_DIR = os.path.join(folder,"logs/")
    os.makedirs(LOGS_DIR,exist_ok=True)
    

    

    model = get_training_model_new(weight_decay,np1=np1,np2=np2,stages=stages)

    from_vgg = dict()
    from_vgg['conv1_1'] = 'block1_conv1'
    from_vgg['conv1_2'] = 'block1_conv2'
    from_vgg['conv2_1'] = 'block2_conv1'
    from_vgg['conv2_2'] = 'block2_conv2'
    from_vgg['conv3_1'] = 'block3_conv1'
    from_vgg['conv3_2'] = 'block3_conv2'
    from_vgg['conv3_3'] = 'block3_conv3'
    from_vgg['conv3_4'] = 'block3_conv4'
    from_vgg['conv4_1'] = 'block4_conv1'
    from_vgg['conv4_2'] = 'block4_conv2'

    # load previous weights or vgg19 if this is the first run
    if os.path.exists(WEIGHTS_BEST):
        print("Loading the best weights...")

        model.load_weights(WEIGHTS_BEST)
        last_epoch = get_last_epoch() + 1
    else:
        print("Loading vgg19 weights...")

        vgg_model = VGG19(include_top=False, weights='imagenet')

        for layer in model.layers:
            if layer.name in from_vgg:
                vgg_layer_name = from_vgg[layer.name]
                layer.set_weights(vgg_model.get_layer(vgg_layer_name).get_weights())
                print("Loaded VGG19 layer: " + vgg_layer_name)

        last_epoch = 0

    # prepare generators


    if use_client_gen:
        train_client = DataGeneratorClient(port=port, host="localhost", hwm=160, batch_size=20,np1=np1,np2=np2,stages=stages)
        train_client.start() # check ds_generator_client.py 
        train_di = train_client.gen()
    
        train_samples = 100

    else:
        
        # Add our augmenter for check stuff

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

    # configure loss functions

 

    losses = {}
    for i in range(1,stages+1):
        losses["weight_stage"+str(i)+"_L1"] = eucl_loss
        losses["weight_stage"+str(i)+"_L2"] = eucl_loss
    print(losses.keys())
   
    # learning rate schedule - equivalent of caffe lr_policy =  "step"
    iterations_per_epoch = train_samples // batch_size
    

    # configure callbacks
    lrate = LearningRateScheduler(step_decay)
    checkpoint = ModelCheckpoint(WEIGHTS_BEST, monitor='loss', verbose=0, save_best_only=False, save_weights_only=True, mode='min', period=1)
    checkpoint2 = ModelCheckpoint(WEIGHTS_100_EPOCH, monitor='loss', verbose=0, save_best_only=False, save_weights_only=True, mode='min', period=100)
    checkpoint3 = ModelCheckpoint(WEIGHTS_COMPLETE, monitor='loss', verbose=0, save_best_only=True,save_weights_only=False, mode='min', period=100)
    csv_logger = CSVLogger(TRAINING_LOG, append=True)
    tb = TensorBoard(log_dir=LOGS_DIR, histogram_freq=0, write_graph=True, write_images=False)

    callbacks_list = [lrate, checkpoint, csv_logger, tb,checkpoint2,checkpoint3]

    # sgd optimizer with lr multipliers
    #multisgd = MultiSGD(lr=base_lr, momentum=momentum, decay=0.0, nesterov=False, lr_mult=lr_mult)
    multisgd = Adam(lr=base_lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # start training
    model.compile(loss=losses, optimizer=multisgd, metrics=["accuracy"])

    model.fit_generator(train_di,
                    steps_per_epoch=train_samples // batch_size,
                    epochs=max_iter,
                    callbacks=callbacks_list,
                    #validation_data=val_di,
                    #validation_steps=val_samples // batch_size,
                    use_multiprocessing=False,
                    initial_epoch=last_epoch
                    )

