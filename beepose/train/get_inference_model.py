import os
import argparse
import json



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', type=int, default =None, help='number of stages')
    parser.add_argument('--gpu',default =1, help= 'what gpu to use, if "all" try to allocate on every gpu'  )
    parser.add_argument('--gpu_fraction', type=float, default =0.9, help= 'how much memory of the gpu to use' )
    parser.add_argument('--np1', type=int, default =None, help= 'Number of pafs' )
    parser.add_argument('--np2', type=int, default =None, help= 'number of heatmaps' )
    parser.add_argument('--config', default=None, type=str, help="Model config json file")
    parser.add_argument('--weights', type=str, required=True ,help =' Path to annotations')
    parser.add_argument('--output', type=str, required=True, help='Output Path' )
    
    
    args = parser.parse_args()
    
    
    config_file = args.config
    
    if config_file is not None:
    
        with open(config_file, 'r') as json_file:
            config = json.load(json_file)

        print(config)

        stages=config["stages"]

        np1=config["np1"]#12 #number of channels for pafs
        np2=config["np2"]#6#number of channels for parts
    else:
        np1 = args.np1
        np2 = args.np2
        stages = args.stages
        
        if np1 is None:
            print("Please provide --np1 option or --config param")
            exit(1)
        if np2 is None:
            print("Please provide --np2 option or --config param")
            exit(1)
        if stages is None:
            print("Please provide --stages option or --config param")
            exit(1)
    
    weights_path = args.weights
    output_path = args.output
    
    gpu = int(args.gpu)
    fraction = float(args.gpu_fraction)

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



    from src.models.train_model import get_testing_model_new

    model = get_testing_model_new(np1=np1, np2=np2, stages=stages)

    model.load_weights(weights_path)
    model.save(output_path)
    
if __name__ == "__main__":
    main()