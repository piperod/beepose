import math
import random

import cv2
import numpy as np
from tensorpack.dataflow.imgaug.geometry import RotationAndCropValid
from enum import Enum 


from skimage.transform import AffineTransform, warp, warp_coords
from math import cos, sin
import numpy as np
from skimage.util import img_as_ubyte



_network_w = 368
_network_h = 368
_scale = 8


def generate_tform(params, translation=True, rotation=True, scale=True):
    '''
    generate_tform(params)
    
    generate AffineTransform from rectangle of size (w,h), such that its center
    end up at `(cx,cy)`, with an angle of `a`, and a scale of `s`
    where params is a dict that contains cx,cy,a,s,w,h

    '''
    
    
    cx = params['cx']
    cy = params['cy']
    a = params['a']
    s = params['s']
    w = params['w']
    h = params['h']
    
    #print(params,'translation:',translation,'rotation:',rotation,'scale:',scale)
    tform = AffineTransform(translation =(0,0))
    

    #coin_toss= np.random.uniform(0,1)
    if translation:
        tform += AffineTransform(translation = (-w/2,-h/2))
    coin_toss= np.random.uniform(0,1)
    if scale:
        #print('scale')
        tform += AffineTransform(scale=(1/s,1/s))
        
    coin_toss= np.random.uniform(0,1)   
    if rotation and coin_toss>0.1:
        #print('rotation')
        tform += AffineTransform(rotation=a)

    
    #tform = AffineTransform(translation = (+w/2,+h/2))
    if translation:
        tform += AffineTransform(translation = (cx,cy))
    return tform


def random_tform_params(imshape,thoraxes, output_shape=(256,256),mina=-np.pi,maxa=np.pi,mins=0.25,maxs=1.2,margin_value=1.5):
    
    imh,imw = imshape
    h,w = output_shape
    
    if ( maxs*np.sqrt(h**2+w**2) >= np.max([imw,imh]) ):
        print("random_tform_params: output_shape and maxs too large, may not fit inside input image")
    
    a=np.random.uniform(mina,maxa)
    
    # Scale is uniform random on a log scale
    #sminlog = -np.log(maxs)
    #smaxlog = np.log(maxs)
    #s=np.exp(np.random.rand(1,1)*(smaxlog-sminlog)+sminlog)[0,0]
    s=np.random.uniform(mins,maxs)
    #coin_toss= np.random.uniform(0,1)
    #if coin_toss>0.5:
    #    s=1
    #print(s)
    # TODO: Multiply by a value to allow objects near the boundary. 
    marginx = np.max(s*np.abs([cos(a)*w/2+sin(a)*h/2,cos(a)*w/2-sin(a)*h/2]))*margin_value
    marginy = np.max(s*np.abs([sin(a)*w/2+cos(a)*h/2,sin(a)*w/2-cos(a)*h/2]))*margin_value
    
    xmin,xmax,ymin,ymax = marginx,imw-marginx, marginy,imh-marginy
    
#     if ((xmin>=xmax) or (ymin>=ymax)):
#         print("random_tform_params: extracted image domain exceed input image size")
    
    cx,cy = random.choice(thoraxes)#(np.random.rand(1,2)*[xmax-xmin,ymax-ymin]+[xmin,ymin]).ravel()

    params = dict(cx=cx,cy=cy,a=a,s=s,w=w,h=h)    
    
    return params


def get_corners(params):
    w = params['w']
    h = params['h']
    return np.array([[0,0],[w,0],[w,h],[0,h]])


def get_input_corners(params):
    p = get_corners(params)
    tform = generate_tform(params)
    p2 = tform(p)
    return p2


def extract_patch(im, params,translation=False,rotation=False,scale=False, tform=None, mode='float64'):
    if (tform is None):
        tform = generate_tform(params,translation=translation,rotation=rotation,scale=scale)
    output_shape = (params['h'],params['w'])
    #print(output_shape)
    if (mode is 'float64'):
        im_out = warp(im, tform, mode='constant',cval=0.5,preserve_range=True, output_shape=output_shape)
        im_out = im_out.astype('uint8')#warp(im, tform, mode='constant', cval=0.5)
    elif (mode is 'int64'):
        im_out = warp(im, tform, mode='constant', output_shape=output_shape, cval=0, preserve_range=True,order=0)
        #im_out = warp(im, tform, mode='constant',  cval=0.5, order=0)
        im_out = im_out.astype('int64')
    else:
        print('nothing done')
    
    cx,cy=int(params['h']//2),int(params['w']//2)
 
    
    #print(im_out.shape)
    return im_out

def transform_coordinates(coords, tform):
    #import pdb;pdb.set_trace()
    coords_out = tform.inverse(coords)
    
    return coords_out

def get_augemented_image_and_label(im, lab, output_shape=(368, 368),mina=-np.pi,maxa=np.pi, translation=True, scale=True, rotation=True, mins=0.25,maxs=1.2, ilumination=0.0):
    # Extract patch for image and labels
    params = random_tform_params(im.shape, output_shape, mins=mins,maxs=maxs)
    tform = generate_tform(params, translation=translation, scale=scale, rotation=rotation)
    im_out = extract_patch(im, params, tform=tform,translation=translation,rotation=rotation,scale=scale)
    lab_out = extract_patch(lab, params, tform=tform,translation=translation,rotation=rotation,scale=scale, mode='int64')
    
    if ilumination > 0.0:
        print('ilumination is on')
        factor = (ilumination) * np.random.random()
        im_out += factor
        im_out = np.clip(im_out, a_min=0.0, a_max=1.0)
    return im_out.reshape(*output_shape, 1), lab_out.reshape(*output_shape, 1)

def get_augmented_image(meta):
    # Todo : Take RNGDataflow to seed the np.random state. 
    # TODO: Center on the bees thorax. 
    
    thoraxes = [ p[2] for p in meta.joint_list if p[2][0]>300 and p[2][0]<2200 and p[2][1]>100 and p[2][1]< 1200]
    #print(meta.height,meta.width)
    params = random_tform_params((meta.height,meta.width),thoraxes, output_shape=meta.output_shape,mina=meta.mina,maxa=meta.maxa,mins=meta.mins,maxs=meta.maxs)
    
    tform = generate_tform(params, translation=meta.translation, scale=meta.scale, rotation=meta.rotation)
    
    
    
    im_out = extract_patch(meta.img, params,
                           tform=tform,translation=meta.translation,rotation=meta.rotation,scale=meta.scale)
    
    mask_all_out = extract_patch(meta.mask_all, params,
                                 tform=tform,translation=meta.translation,rotation=meta.rotation,scale=meta.scale,mode='int64')
    mask_miss_out = extract_patch(meta.mask_miss, params,
                                  tform=tform,translation=meta.translation,rotation=meta.rotation,scale=meta.scale,mode='int64')
    
    new_joint_list=[]
    for joints in meta.joint_list:
        joints = np.array([[j[0],j[1]] for j in joints])
        #print(joints)
        lab_out = transform_coordinates(joints,tform)
        new_joint_list.append(lab_out)
       
    meta.img_org = meta.img  
    meta.img = im_out
    meta.mask_all_out = mask_all_out
    meta.mask_miss_out = mask_miss_out
    meta.joint_list =new_joint_list
    meta.height,meta.width = meta.output_shape
    meta.params = params
   
    return meta
    
def set_network_input_wh(w, h):
    global _network_w, _network_h
    _network_w, _network_h = w, h
     

def set_network_scale(scale):
    global _scale
    _scale = scale

def check_network_input_wh():
    global _network_w, _network_h,_scale
    return _network_w, _network_h, _scale

def pose_to_img(meta_l):
    global _network_w, _network_h, _scale
    
    try:
        target_size=(_network_w // _scale, _network_h // _scale)
        #print( _network_w, _network_h, _scale)
        if meta_l[0].is_train:
            if meta_l[0].image_type == 'RGB':
                resized =meta_l[0].img#cv2.resize(meta_l[0].img,(_network_w , _network_h),interpolation=cv2.INTER_CUBIC)
            else:
                resized = meta_l[0].img[:,:,[2,1,0]]
            # Masking in the shape of heatmaps and vectormaps to create weights
       
            mask_img = cv2.resize(meta_l[0].mask_miss_out,target_size,interpolation=cv2.INTER_NEAREST)

            #mask_miss = meta_l[0].mask_miss_out#cv2.resize(meta_l[0].mask_miss_out,target_size)

            # mask - the same for vec_weights, heat_weights
       
            vec_weights = np.repeat(mask_img[:,:,np.newaxis], meta_l[0].numparts*2, axis=2)
            heat_weights=  np.repeat(mask_img[:,:,np.newaxis], meta_l[0].numparts, axis=2)
            #print(meta_l[0].params)
            return [
            resized,heat_weights,vec_weights,
            meta_l[0].get_heatmap(target_size=target_size),
            meta_l[0].get_vectormap(target_size=target_size)]
        else:
            #print(meta_l[0].img.shape)
            
            if meta_l[0].image_type == 'RGB':
                
                w,h= meta_l[0].img.shape[0]//4, meta_l[0].img.shape[1]//4
                resized =cv2.resize(meta_l[0].img,(h,w),interpolation=cv2.INTER_CUBIC)
                #resized = resized[:_network_w,:_network_h]
            else:
                resized = meta_l[0].img[:,:,[2,1,0]]
            target_size1=(w // _scale, h // _scale,meta_l[0].numparts*2)
            target_size2=(w // _scale, h // _scale,meta_l[0].numparts)
            return [
            resized,np.ones(target_size2),np.ones(target_size1)]
    except:
        
        if meta_l[0].image_type == 'RGB':
            w,h= meta_l[0].img.shape[0], meta_l[0].img.shape[1]
            resized =cv2.resize(meta_l[0].img,(h//4,w//4),interpolation=cv2.INTER_CUBIC)
            #resized = resized[:_network_w,:_network_h]
        else:
            resized = meta_l[0].img[:,:,[2,1,0]]
        if meta_l[0].is_train:
            debug = True
        else:
            debug = False
        return [
        resized,
        meta_l[0].get_heatmap(target_size=(_network_w // _scale, _network_h // _scale)),
        meta_l[0].get_vectormap(target_size=(_network_w // _scale, _network_h // _scale),debug=debug)
    ]

