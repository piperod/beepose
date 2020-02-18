import keras 
import tfmpl
import tensorflow as tf
from keras import backend as K
import numpy as np
import cv2
class AttentionLogger(keras.callbacks.Callback):
        def __init__(self, val_data, logsdir,freq=10):
            super(AttentionLogger, self).__init__()
            self.logsdir = logsdir  # where the event files will be written 
            self.validation_data = val_data # validation data generator
            self.writer = tf.summary.FileWriter(self.logsdir)  # creating the summary writer
            self.freq = freq
        @tfmpl.figure_tensor
        def attention_matplotlib(self, gen_images,gen_heatmaps,gen_vectormaps): 
            '''
            Creates a matplotlib figure and writes it to tensorboard using tf-matplotlib
            gen_images: The image tensor of shape (batchsize,width,height,channels) you want to write to tensorboard
            '''  
            r, c = 2,5 # want to write 25 images as a 5x5 matplotlib subplot in TBD (tensorboard)
            figs = tfmpl.create_figures(1, figsize=(10,15))
            cnt = 0
            for idx, f in enumerate(figs):
                for i in range(r):
                    for j in range(c):    
                        ax = f.add_subplot(r,c,cnt+1)
                        ax.set_yticklabels([])
                        ax.set_xticklabels([])
                        #import pdb;pdb.set_trace()
                        img_shape =gen_images[0][cnt].shape
                        hmps= np.squeeze(gen_heatmaps[cnt])
                        hmps =cv2.resize(hmps, (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
                        hmp = cv2.resize(hmps,(img_shape[1],img_shape[0]),interpolation=cv2.INTER_CUBIC)
                        ax.imshow(gen_images[0][cnt])
                        ax.imshow(hmp[:,:,cnt%5],alpha=0.5)
                        #ax.imshow(hmp[:,:,1],alpha=0.5)
                        #ax.imshow(hmp[:,:,2],alpha=0.5)
                        #ax.imshow(hmp[:,:,3],alpha=0.5)
                        #ax.imshow(hmp[:,:,4],alpha=0.5)
                        # writes the image at index cnt to the 5x5 grid
                        
                        cnt+=1
                f.tight_layout()
            return figs

        def on_train_begin(self, logs=None):  # when the training begins (run only once)
                image_summary = [] # creating a list of summaries needed (can be scalar, images, histograms etc)
                for index in range(len(self.model.output)):  # self.model is accessible within callback
                    img_sum = tf.summary.image('img{}'.format(index), self.attention_matplotlib(self.model.output[index]))                    
                    image_summary.append(img_sum)
                self.total_summary = tf.summary.merge(image_summary)

        def on_epoch_end(self, epoch, logs = None):   # at the end of each epoch run this
            if epoch%self.freq ==0:
                logs = logs or {} 
                x = next(self.validation_data)  # get data from the generator
                output = self.model.predict(x)
                img_sum = tf.summary.image('preds{}'.format(epoch), self.attention_matplotlib(x,output[1],output[0]))
                sess_img =K.get_session().run(img_sum)
                self.writer.add_summary(sess_img)
            #import pdb;pdb.set_trace()
                
            # get the backend session and sun the merged summary with appropriate feed_dict
            #sess_run_summary = K.get_session().run(self.total_summary, feed_dict = {self.model.input: np.array(x)})
            #self.writer.add_summary(sess_run_summary, global_step =epoch)  #finally write the summary!