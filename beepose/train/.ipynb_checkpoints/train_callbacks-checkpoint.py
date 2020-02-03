import keras
from skimage.io import imread
import os 
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from datetime import datetime 
import numpy as np
import io
import inspect


class TensorBoardImage(keras.callbacks.Callback):
    
    def __init__(self, val_dir,log_dir,scale=0.25):
        super().__init__() 
     
        self.val_dir = val_dir
        self.log_dir =log_dir+'/plots/' + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.val_imgs =  glob.glob(os.path.join(val_dir,'*.*p*'))
        self.scale = scale
        
        
    def on_epoch_end(self, epoch, logs={}):
        print('Performing validation step ')
       
        #file_writer = tf.summary.create_file_writer(self.logdir)
        
        file_writer = tf.summary.FileWriter(self.log_dir)
        # Load image
        images = []
        for img in self.val_imgs[:5]:
            image = imread(img)
            image = cv2.resize(image,(0,0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
            images.append(image)
            input_image = np.transpose(np.float32(image[:,:,:,np.newaxis]), (3,0,1,2))
            heat_image = np.ones((1,1,1,6))
            vec_image = np.ones((1,1,1,12))
            output_blobs = self.model.predict([input_image,vec_image,heat_image])
            
            heatmap = np.squeeze(output_blobs[1]) # output 1 is heatmaps
            heatmap = cv2.resize(heatmap[:,:,2], (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
            
            plt.imshow(image)
            plt.imshow(heatmap)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            #image = buf.getvalue()
            images.append(image)
            #image = tf.Summary.Image(img,image)
        print(inspect.getsource(tf.summary.image))   
        images = np.array(images)
        #with file_writer.as_default():
        #image = tf.Summary.Image('validation images',images,max_outputs=5,step=0)
        #image = tf.Summary.Image('something.png',image,max_outputs=3)   
        input_image = tf.Summary(value=[tf.Summary.Image(tag='validation-images', image=image)])
            
        writer.add_summary(image, epoch)
            
            
            #image = tf.image.decode_png(buf.getvalue(), channels=3)
            #heatmap = tf.Summary(value=[tf.Summary.Value(tag='heatmap_'+img, image=image)])
            
           
            
        
        writer.close()

        return

