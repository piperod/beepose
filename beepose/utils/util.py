import numpy as np
from io import StringIO
import PIL.Image
import pandas as pd 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import json
import glob,os
import cv2
from scipy.optimize import linear_sum_assignment as hungarian
from keras.models import model_from_json
 
def merge_solutions_by_batch(results):
    
    final = dict()
    for r in results:
        final.update(r)
    return final 

def read_json(filename):
    with open(filename) as file:
        detections= json.load(file)
    return detections

def save_json(path,obj):
    with open(path,'w') as file:
        json.dump(obj,file,cls=NumpyEncoder)


def load_model(json_file,weights_file):
	loaded_model = load_json(json_file)
	model=model_from_json(loaded_model)
	model.load_weights(weights_file)
	return model

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj,np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj) 

    
def rotate_bound2(image,x,y,angle, w,h):
    # grab the dimensions of the image and then determine the
    # center
    (h0, w0) = image.shape[:2]
    (pX, pY) = (x, y) # Rect center in input
    
    (cX, cY) = (w / 2, h / 2)     # Rect center in output
    
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0) # angle in degrees
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
  # adjust the rotation matrix to take into account translation
    M[0, 2] += pX - cX
    M[1, 2] += pY - cY
    
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (w, h), flags=cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)

def distance_point(dt,gt,tl=1):
    return ((dt[0]-gt[0])**2+(dt[1]-gt[1])**2)/tl

def distance_line_point(m,point):
    import numpy as np
    x1 =m[0][0]
    y1 =m[0][1]
    x2 =m[1][0]
    y2 =m[1][1]
    numerator = abs((y2-y1)*point[0]-(x2-x1)*point[1]+x2*y1-y2*x1)
    denominator = np.sqrt((y2-y1)**2+(x2-x1)**2)
    return numerator/denominator

def showBGRimage(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    a[:,:,[0,2]] = a[:,:,[2,0]] # for B,G,R order
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

def showmap(a, fmt='png'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

#def checkparam(param):
#    octave = param['octave']
#    starting_range = param['starting_range']
#    ending_range = param['ending_range']
#    assert starting_range <= ending_range, 'starting ratio should <= ending ratio'
#    assert octave >= 1, 'octave should >= 1'
#    return starting_range, ending_range, octave

def getJetColor(v, vmin, vmax):
    c = np.zeros((3))
    if (v < vmin):
        v = vmin
    if (v > vmax):
        v = vmax
    dv = vmax - vmin
    if (v < (vmin + 0.125 * dv)): 
        c[0] = 256 * (0.5 + (v * 4)) #B: 0.5 ~ 1
    elif (v < (vmin + 0.375 * dv)):
        c[0] = 255
        c[1] = 256 * (v - 0.125) * 4 #G: 0 ~ 1
    elif (v < (vmin + 0.625 * dv)):
        c[0] = 256 * (-4 * v + 2.5)  #B: 1 ~ 0
        c[1] = 255
        c[2] = 256 * (4 * (v - 0.375)) #R: 0 ~ 1
    elif (v < (vmin + 0.875 * dv)):
        c[1] = 256 * (-4 * v + 3.5)  #G: 1 ~ 0
        c[2] = 255
    else:
        c[2] = 256 * (-4 * v + 4.5) #R: 1 ~ 0.5                      
    return c

def colorize(gray_img):
    out = np.zeros(gray_img.shape + (3,))
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            out[y,x,:] = getJetColor(gray_img[y,x], 0, 1)
    return out

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad






def distance(dt,gt,tl=1):
    return ((dt[0]-gt[0])**2+(dt[1]-gt[1])**2)/tl

def distance_tracks(dt,gt,tl=1):
    return (np.sqrt((dt[0]-gt[0])**2+(dt[1]-gt[1])**2))/tl

def cost_matrix_tracks(ground_t,detections,threshold):
    total = len(ground_t)+len(detections)
    cost_m = np.zeros((total,total))
    for i in range(total):
        for j in range(total):
            if i < len(ground_t) and j <len(detections):
                cost_m[i][j] = distance_tracks(ground_t[i],detections[j])
            else:
                cost_m[i][j] = threshold
    
    return cost_m

def distance_tracks_wb(dt,gt,mapping):
    d=[]
    d.append(np.sqrt((dt[0]-gt[0])**2+(dt[1]-gt[1])**2))
    for m in mapping: 
        if gt==m[0]:
            d.append(np.sqrt((m[0][0]-gt[0])**2+(m[0][1]-gt[1])**2))
        elif gt==m[1]:
            d.append(np.sqrt((m[0][0]-gt[0])**2+(m[0][1]-gt[1])**2))
    if len(d)==1:
        d.append(300)
    elif len(d)==2:
        d.append(100)
    return np.array(d).mean()
            
def cost_matrix_tracks_wb(ground_t,detections,new_mapping,threshold):
    total = len(ground_t)+len(detections)
    cost_m = np.zeros((total,total))
    for i in range(total):
        for j in range(total):
            if i < len(ground_t) and j <len(detections):
                cost_m[i][j] = distance_tracks_wb(ground_t[i],detections[j],new_mapping)
            else:
                cost_m[i][j] = threshold
    
    return cost_m

def cost_matrix(ground_t,detections,threshold):
    total = len(ground_t)+len(detections)
    cost_m = np.zeros((total,total))
    for i in range(total):
        for j in range(total):
            if i < len(ground_t) and j <len(detections):
                cost_m[i][j] = distance(ground_t[i],detections[j],len(ground_t))
            else:
                cost_m[i][j] = threshold
    
    return cost_m

def cost_matrix_mappings(ground_mappings,mapings,threshold):
    total = len(ground_mappings)+len(mapings)
    cost_m = np.zeros((total,total))
    for i in range(total):
        for j in range(total):
            if i < len(ground_mappings) and j <len(mapings):
                cost_m[i][j] = (distance(ground_mappings[i][0],mapings[j][0],len(ground_mappings)) + distance(ground_mappings[i] [1],mapings[j][1],len(ground_mappings)))/2
            else:
                cost_m[i][j] = threshold*2
    
    return cost_m
def dfs(graph, start):

    visited, stack = set(), [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(graph[vertex] - visited)
    return visited

def clean_parts(parts,mappings):
    new_parts=[]
    excluded=set()
    map1 = [tuple(m[0]) for m in mappings]
    map2 = [tuple(m[1]) for m in mappings]
    mapt = map1+map2
    for i in range(len(parts)):    
        for j in range(len(parts)):
            d=distance_tracks(parts[i],parts[j])
            if d ==0 and tuple(parts[j]) not in excluded:
                new_parts.append(parts[j])
            elif tuple(parts[j]) not in mapt:
                excluded.add(tuple(parts[j]))
            #if d>0 and d<4 and tuple(parts[j]) not in excluded:
            #    excluded.add(tuple(parts[j]))
                
    return new_parts
                
def clean_detections(detections):
    keylist= list(detections.keys())
    for i in range(len(keylist)):
        detections[keylist[i]]['new_mapping']=[]
        for mapping in detections[keylist[i]]['mapping']:
            new_mapping=[[mapping[1][0],mapping[0][0]],[mapping[1][1],mapping[0][1]],mapping[2],mapping[3]]
            detections[keylist[i]]['new_mapping'].append(new_mapping)
        for k in detections[keylist[i]]['parts'].keys():
            new_parts = clean_parts(detections[keylist[i]]['parts'][k],detections[keylist[i]]['new_mapping'])
            detections[keylist[i]]['parts'][k]=new_parts
        
    return detections 

    
import math 
def find_angle(part,mapping,typ=[1,3]):
    for m in mapping:
        if m[1]==part[:2]:
            myradians = math.atan2(m[0][0]-m[1][0],m[0][1]-m[1][1])
            mydegrees = math.degrees(myradians)
            return (mydegrees-90)%360
    return -1 





def cost_matrix_mappings(ground_mappings,mapings,threshold):
    total = len(ground_mappings)+len(mapings)
    cost_m = np.zeros((total,total))
    for i in range(total):
        for j in range(total):
            if i < len(ground_mappings) and j <len(mapings):
                cost_m[i][j] = (distance(ground_mappings[i][0],mapings[j][0],len(ground_mappings)) + distance(ground_mappings[i] [1],mapings[j][1],len(ground_mappings)))/2
            else:
                cost_m[i][j] = threshold*2
    
    return cost_m
def dfs(graph, start):

    visited, stack = set(), [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(graph[vertex] - visited)
    return visited

def clean_parts(parts,mappings):
    new_parts=[]
    excluded=set()
    map1 = [tuple(m[0]) for m in mappings]
    map2 = [tuple(m[1]) for m in mappings]
    mapt = map1+map2
    for i in range(len(parts)):    
        for j in range(len(parts)):
            d=distance_tracks(parts[i],parts[j])
            if d ==0 and tuple(parts[j]) not in excluded:
                new_parts.append(parts[j])
            elif tuple(parts[j]) not in mapt:
                excluded.add(tuple(parts[j]))
            #if d>0 and d<4 and tuple(parts[j]) not in excluded:
            #    excluded.add(tuple(parts[j]))
                
    return new_parts
                
def clean_detections(detections):
    keylist= list(detections.keys())
    for i in range(len(keylist)):
        detections[keylist[i]]['new_mapping']=[]
        for mapping in detections[keylist[i]]['mapping']:
            new_mapping=[[mapping[1][0],mapping[0][0]],[mapping[1][1],mapping[0][1]],mapping[2],mapping[3]]
            detections[keylist[i]]['new_mapping'].append(new_mapping)
        for k in detections[keylist[i]]['parts'].keys():
            new_parts = clean_parts(detections[keylist[i]]['parts'][k],detections[keylist[i]]['new_mapping'])
            detections[keylist[i]]['parts'][k]=new_parts
        
    return detections 


def is_skeleton_zero_based(skeleton):
    for limb in skeleton:
        for part in limb:
            if part == 0:
                return True
    return False

def to_one_based_skeleton(skeleton):
    one_based_skeleton = list()
    
    for part_a, part_b in skeleton:
        one_based_skeleton.append(part_a + 1, part_b + 1)
        
    return one_based_skeleton
    

def one_index_based_skeleton(skeleton):
    if is_skeleton_zero_based(skeleton):
        skeleton = to_one_based_index(skeleton)
    return skeleton


def get_skeleton_from_json(filename):
    """
    Get the numparts and skeleton from the coco format json file.
    Skeleton should be 1 based index.  
    Input:
        filename: str 
    Output:
        numparts: int
        skeleton: list
    """
    data =  read_json(filename)
    pose_info = data["categories"]

    beepose_info = pose_info[0]
    
    numparts = len(beepose_info["keypoints"])
    skeleton = beepose_info["skeleton"]
    
    # detect 0-based skeleton and change to one-based index
    # if it is necessary.
    skeleton = one_index_based_skeleton(skeleton)
    
    return numparts, skeleton


def get_skeleton_mapIdx(numparts):
    """
    Calculate the mapsIdx for each limb from the skeleton.
    Input:
        skeleton: list of part connection 
    Output:
        list of ids for x and y for part
    """
    connections_num = numparts
    mapIdx = list()
    
    for i in range(connections_num):
        mapIdx.append([2 * i, (2 * i) + 1])
    return mapIdx
    
# NONMAXIMA SUPPRESSION FROM https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/
# import the necessary packages
import numpy as np


def boxes2dets(boxes,size=20):
    dets=[]
    for b in boxes:
        dets.append([b[0]+size,b[1]+size,b[-1]])
    return dets
def dets2boxes(parts,size=20):
    boxes=[]
    for p in parts:
        boxes.append([p[0]-size,p[1]-size,p[0]+size,p[1]+size,p[2]])
    return np.array(boxes)
#  Felzenszwalb et al.
def non_max_suppression_slow(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# initialize the list of picked indexes
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list, add the index
		# value to the list of picked indexes, then initialize
		# the suppression list (i.e. indexes that will be deleted)
		# using the last index
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]
		# loop over all indexes in the indexes list
		for pos in range(0, last):
			# grab the current index
			j = idxs[pos]
 
			# find the largest (x, y) coordinates for the start of
			# the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box
			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])
 
			# compute the width and height of the bounding box
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)
 
			# compute the ratio of overlap between the computed
			# bounding box and the bounding box in the area list
			overlap = float(w * h) / area[j]
 
			# if there is sufficient overlap, suppress the
			# current bounding box
			if overlap > overlapThresh:
				suppress.append(pos)
 
		# delete all indexes from the index list that are in the
		# suppression list
		idxs = np.delete(idxs, suppress)
 
	# return only the bounding boxes that were picked
	return boxes[pick]

def boxes2peaks(boxes,size=15):
    dets=[]
    for b in boxes:
        dets.append((b[0]+size,b[1]+size))
    return dets
def peaks2boxes(parts,size=15):
    boxes=[]
    for p in parts:
        boxes.append([p[0]-size,p[1]-size,p[0]+size,p[1]+size])
    return np.array(boxes)


def non_max_suppression_op(peaks,overlap=0.6,size=15):
    boxes= non_max_suppression_fast(peaks2boxes(peaks,size),overlap)
    dets = boxes2peaks(boxes,size)
    return dets
        
# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
 
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
 
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")


Colors= [
'#F0F8FF',
'#FAEBD7',
'#00FFFF',
'#7FFFD4',
'#F0FFFF',
'#F5F5DC',
'#FFE4C4',
'#000000',
'#FFEBCD',
'#0000FF',
'#8A2BE2',
'#A52A2A',
'#DEB887',
'#5F9EA0',
'#7FFF00',
'#D2691E',
'#FF7F50',
'#6495ED',
'#FFF8DC',
'#DC143C',
'#00FFFF',
'#00008B',
'#008B8B',
'#B8860B',
'#A9A9A9',
'#006400',
'#BDB76B',
'#8B008B',
'#556B2F',
'#FF8C00',
'#9932CC',
'#8B0000',
'#E9967A',
'#8FBC8F',
'#483D8B',
'#2F4F4F',
'#00CED1',
'#9400D3',
'#FF1493',
'#00BFFF',
'#696969',
'#1E90FF',
'#B22222',
'#FFFAF0',
'#228B22',
'#FF00FF',
'#DCDCDC',
'#F8F8FF',
'#FFD700',
'#DAA520',
'#808080',
'#008000',
'#ADFF2F',
'#F0FFF0',
'#FF69B4',
'#CD5C5C',
'#4B0082',
'#FFFFF0',
'#F0E68C',
'#E6E6FA',
'#FFF0F5',
'#7CFC00',
'#FFFACD',
'#ADD8E6',
'#F08080',
'#E0FFFF',
'#FAFAD2',
'#90EE90',
'#D3D3D3',
'#FFB6C1',
'#FFA07A',
'#20B2AA',
'#87CEFA',
'#778899',
'#B0C4DE',
'#FFFFE0',
'#00FF00',
'#32CD32',
'#FAF0E6',
'#FF00FF',
'#800000',
'#66CDAA',
'#0000CD',
'#BA55D3',
'#9370DB',
'#3CB371',
'#7B68EE',
'#00FA9A',
'#48D1CC',
'#C71585',
'#191970',
'#F5FFFA',
'#FFE4E1',
'#FFE4B5',
'#FFDEAD',
'#000080',
'#FDF5E6',
'#808000',
'#6B8E23',
'#FFA500',
'#FF4500',
'#DA70D6',
'#EEE8AA',
'#98FB98',
'#AFEEEE',
'#DB7093',
'#FFEFD5',
'#FFDAB9',
'#CD853F',
'#FFC0CB',
'#DDA0DD',
'#B0E0E6',
'#800080',
'#FF0000',
'#BC8F8F',
'#4169E1',
'#8B4513',
'#FA8072',
'#FAA460',
'#2E8B57',
'#FFF5EE',
'#A0522D',
'#C0C0C0',
'#87CEEB',
'#6A5ACD',
'#708090',
'#FFFAFA',
'#00FF7F',
'#4682B4',
'#D2B48C',
'#008080',
'#D8BFD8',
'#FF6347',
'#40E0D0',
'#EE82EE',
'#F5DEB3',
'#FFFFFF',
'#F5F5F5',
'#FFFF00',
'#9ACD32']
