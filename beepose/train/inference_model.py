import os
import argparse

from beepose.models.train_model import get_testing_model_new
 

def save_inference_model(stages,np1,np2,weights_path,output_path):
    
    model = get_testing_model_new(np1=np1, np2=np2, stages=stages)
    model.load_weights(weights_path)
    model.save(output_path)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', type=int, default =6, help='number of stages')
    parser.add_argument('--gpu',default =1, help= 'what gpu to use, if "all" try to allocate on every gpu'  )
    parser.add_argument('--gpu_fraction', type=float, default =0.9, help= 'how much memory of the gpu to use' )
    parser.add_argument('--np1', type=int, default =12, help= 'Number of pafs' )
    parser.add_argument('--np2', type=int, default =6, help= 'number of heatmaps' )
    parser.add_argument('--weights', type=str, required=True ,help =' Path to annotations')
    parser.add_argument('--output', type=str, required=True, help='Output Path' )
    
    
    args = parser.parse_args()
    
    stages=int(args.stages)
    fraction = float(args.gpu_fraction)
    np1=int(args.np1)#12 #number of channels for pafs
    np2=int(args.np2)#6#number of channels for parts
    weights_path = args.weights
    output_path = args.output
    numparts = np2 
    gpu = int(args.gpu)

    print(gpu)
    if gpu != 'all':
        print(gpu)
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]="%d"%gpu
        
    import keras.backend as K
    import tensorflow as tf 
    
    
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = fraction
    session = tf.Session(config=config)

    save_inference_model(stages,np1,np2,weights_path,output_path)

    