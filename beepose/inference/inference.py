import glob,os,sys
sys.path.append('..')
import cv2
import math
import time
from beepose.utils import util 
import numpy as np
import json 
from scipy.ndimage.filters import gaussian_filter
import logging
logger = logging.getLogger(__name__)
import numba
FPS=20

# Color constant
colors= [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

@numba.jit(nopython=True, parallel=True)
def calculate_peaks(numparts,heatmap_avg):
    #Right now there is a score for every part since some parts are likely to need lower thresholds. 
    # TODO: Run grid search to find the ideal values. 
    score=[0.2,0.2,0.2,0.2,0.2,0.5,0.5,0.5,0.5]
    all_peaks = []
    peak_counter = 0
    if len(score)<numparts:
        score = score[:numparts]
        ##logger.ERROR('Not enough scores provided for number of parts')
        #return
    #threshold_detection = params['thre1']
    #tic_localmax=time.time()
    for part in range(numparts):
        map_ori = heatmap_avg[:, :, part]
        map = map_ori
        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]
        peaks_binary = np.logical_and(np.logical_and(np.logical_and(map >= map_left, map >= map_right), np.logical_and(map >= map_up, map >= map_down)), map >score[part])
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score_and_id = [ x + (map_ori[x[1], x[0]], i+peak_counter,) for i,x in enumerate(peaks)] #if x[0]>0 and x[1]>0 ]
        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)
    return all_peaks

def candidate_selection(mapIdx,limbSeq,paf_avg,distance_tolerance,resize,thre2,width_ori):
    connection_all = []
    special_k = []
    mid_num = 20
    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x  for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    # failure case when 2 body parts overlaps
                    if norm == 0:
                        continue
                    if norm >distance_tolerance//resize:
                        continue 
                        
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                         for I in range(len(startend))])
                    vec_y = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                         for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * width_ori / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > thre2)[0]) > 0.7 * len(
                        score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior,
                                                     score_with_dist_prior + candA[i][2] + candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])
    return 
    
def inference(input_image,model, params, model_params,show=True,np1=19,np2=38,resize=1,
                distance_tolerance=310,numparts=5,
                mapIdx=[[0,1],[2,3],[4,5],[6,7],[8,9]],
                limbSeq=[[1,3],[3,2],[2,4],[2,5],[1,2]],
                image_type='RGB'):
    """
    This function uses the model to generate the heatmaps and pafs then use them to produce the poses. 
    
    inputs: 
    
        - input_image : An image
        - model : A trained keras model 
        - params : Parameters used  for adapting the image to match training
        - model_params : Parameters for padding the images after resizing
        - show : Boolean to generate a canvas with the poses on there. 
        - np1 : Number of channels for pafs. 
        - np2 : Number of channels for heatmaps. 
        - resize: Resize factor of the image. 
        - distance_tolerance: Maximum distance between two parts. 
        - numparts: Number of parts
        - mapIdx: configuration for the pafs  0 based 
        - limbSeq: configuration of the poses. It should match with the pafs configuration. 1 based
        - image_type: How was trained the model with RGB or BGR images. 
        
    Outputs : 
        - canvas: if Show, generates an image with the pose. 
        - mapping : How the parts are connected. 
        - parts : Detections for each of the parts considered. 
        
        model_params['boxsize'] 
        model_params['stride']
        model_params['padValue']
        params['scale_search']
        params['thre1']
        params['thre2']
    """
    
    profiling ={}
    tic_initialize=time.time()
    if image_type=='RGB':
        oriImg = cv2.cvtColor(input_image,cv2.COLOR_RGB2BGR)#cv2.imread(input_image)  # B,G,R order
    else: 
        oriImg = input_image
    canvas = oriImg.copy()#cv2.imread(input_image) 
    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in params['scale_search']]

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], np1))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], np2))
    

    scale =1
    imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'],
                                                          model_params['padValue'])
    input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)
    toc_initialize = time.time()
    logger.debug('Initilizing frame time is %.5f' % (toc_initialize - tic_initialize))
    tic_predict=time.time()
    output_blobs = model.predict(input_img)
    toc_predict=time.time()
    logger.debug('predict frame time is %.5f' % (toc_predict - tic_predict))
        
    # extract outputs, resize, and remove padding
    tic_resizing = time.time()
    heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
    heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],interpolation=cv2.INTER_CUBIC)
    heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],:]
    heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
    paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
    paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
    paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
    paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        
    heatmap_avg =  heatmap[...] +heatmap_avg #/ len(multiplier)
    paf_avg =  paf[...] +paf_avg# / len(multiplier)
    toc_resizing = time.time()
    logger.debug('Resizing prediction frame time is %.5f' % (toc_resizing - tic_resizing))

    #all_peaks = []
    #peak_counter = 0
    #threshold_detection = params['thre1']
    tic_localmax=time.time()
    # New function to allow parralel execution
    all_peaks=calculate_peaks(numparts,heatmap_avg)
    #print(all_peaks)
    
    toc_localmax=time.time()
    logger.debug('Non Local maxima frame time is %.5f' % (toc_localmax - tic_localmax))
    connection_all = []
    special_k = []
    mid_num = 15
    tic_candidate= time.time()
    
    
    for k in range(len(mapIdx)):
        score_mid = paf_avg[:, :, [x  for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    # failure case when 2 body parts overlaps
                    if norm == 0:
                        continue
                    if norm >distance_tolerance//resize:
                        continue 
                        
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                         for I in range(len(startend))])
                    vec_y = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                         for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > 0.7 * len(
                        score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior,
                                                     score_with_dist_prior + candA[i][2] + candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])
    toc_candidate = time.time()
    logger.debug('Candidate frame time is %.5f' % (toc_candidate - tic_candidate))
   
    # last number in each row is the total parts number of that animal
    # the second last number in each row is the score of the overall configuration
    
    candidate = np.array([item for sublist in all_peaks for item in sublist])
    print(len(candidate))
    subset = -1 * np.ones((0, len(candidate)+1))
    tic_pafscore=time.time()
    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1
                
                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < numparts:
                    row = -1 * np.ones(len(candidate)+1)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + \
                              connection_all[k][i][2]
                    subset = np.vstack([subset, row])
    toc_pafscore=time.time()
    logger.debug('Paf scoring frame time is %.5f' % (toc_pafscore - tic_pafscore))
    # delete some rows of subset which has few parts occur
    tic_parsing =time.time()
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 2 or subset[i][-2] / subset[i][-1] < 0.2:
             deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)
    temp_parts ={}
    parts={}
    for i in range(numparts):#17
        temp_parts[i]=[]
        for j in range(len(all_peaks[i])):
            a=all_peaks[i][j][0]*resize
            b=all_peaks[i][j][1]*resize
            c=all_peaks[i][j][2]
            temp_parts[i].append([int(a),int(b),c])
        parts[i]=temp_parts[i]
    mappings=[]
    for i in range(numparts):#17
        for n in range(len(subset)):
            kind=limbSeq[i]
            index = subset[n][np.array(kind) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            S = candidate[index.astype(int), 2]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0]*resize - X[1]*resize) ** 2 + (Y[0]*resize - Y[1]*resize) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0]*resize - X[1]*resize, Y[0]*resize - Y[1]*resize))
            mappings.append([[int(Y[0])*resize,int(X[0])*resize],[int(Y[1])*resize,int(X[1])*resize],np.array(S).mean(),length,angle,kind])
    toc_parsing =time.time()
    logger.debug('Parsing result frame time is %.5f' % (toc_parsing - tic_parsing))
    if show:
        size=1
        thick=-1
        for i in range(numparts):#17
            if i > 4 and i<7:
                size=4
                thick =1
            if i>6:
                size=4
                thick =3
            for j in range(len(all_peaks[i])):
                
                cv2.circle(canvas, all_peaks[i][j][0:2], size, colors[i], thickness=thick)

        stickwidth = 10//(resize-1) #4

        for i in range(numparts):#17
            for n in range(len(subset)):
                index = subset[n][np.array(limbSeq[i]) - 1]
                if -1 in index:
                    continue
                cur_canvas = canvas.copy()
                Y = candidate[index.astype(int), 0]
                X = candidate[index.astype(int), 1]
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0,
                                           360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    
    return canvas,mappings,parts


def inference_by_batch (input_imgs,ori,hmps,pafs,init,params,model_params,show=False,resize=4,np1=6,np2=12,numparts=5,mapIdx=[[0,1],[2,3],[4,5],[6,7],[8,9]],
                        limbSeq=[[1,3],[3,2],[2,4],[2,5],[1,2]],distance_tolerance = 300):
    frame_detections={}
    canvas_out=[]
    for idx in range(len(input_imgs)):
        frame_detections[idx+init]={}
        
        pad =[0,0,0,0]
        oriImg = ori[idx]
        
        canvas = input_imgs[idx].copy()
        imageToTest_padded = canvas
        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], np1))
        paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], np2))
        heatmap = np.squeeze(hmps[idx])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],:]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
        paf = np.squeeze(pafs[idx])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
            
        heatmap_avg =  heatmap[...] +heatmap_avg #/ len(multiplier)
        paf_avg =  paf[...] +paf_avg# / len(multiplier)
        
    
        all_peaks = []
        peak_counter = 0
        for part in range(numparts):
            map_ori = heatmap_avg[:, :, part]
            map = gaussian_filter(map_ori, sigma=3)
    
            map_left = np.zeros(map.shape)
            map_left[1:, :] = map[:-1, :]
            map_right = np.zeros(map.shape)
            map_right[:-1, :] = map[1:, :]
            map_up = np.zeros(map.shape)
            map_up[:, 1:] = map[:, :-1]
            map_down = np.zeros(map.shape)
            map_down[:, :-1] = map[:, 1:]
    
            peaks_binary = np.logical_and.reduce(
                (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
            id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]
    
            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)
        connection_all = []
        special_k = []
        mid_num = 4
        for k in range(len(mapIdx)):
            score_mid = paf_avg[:, :, [x  for x in mapIdx[k]]]
            candA = all_peaks[limbSeq[k][0] - 1]
            candB = all_peaks[limbSeq[k][1] - 1]
            nA = len(candA)
            nB = len(candB)
            indexA, indexB = limbSeq[k]
            if (nA != 0 and nB != 0):
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        # failure case when 2 body parts overlaps
                        if norm == 0:
                            continue
                        if norm >distance_tolerance//resize:
                            continue 
                            
                        vec = np.divide(vec, norm)
    
                        startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                       np.linspace(candA[i][1], candB[j][1], num=mid_num)))
    
                        vec_x = np.array(
                            [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                             for I in range(len(startend))])
                        vec_y = np.array(
                            [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                             for I in range(len(startend))])
    
                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                            0.5 * oriImg.shape[0] / norm - 1, 0)
                        criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > 0.8 * len(
                            score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append([i, j, score_with_dist_prior,
                                                         score_with_dist_prior + candA[i][2] + candB[j][2]])
    
                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if (i not in connection[:, 3] and j not in connection[:, 4]):
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if (len(connection) >= min(nA, nB)):
                            break
    
                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])
        # last number in each row is the total parts number of that animal
        # the second last number in each row is the score of the overall configuration
        
        candidate = np.array([item for sublist in all_peaks for item in sublist])
        subset = -1 * np.ones((0, len(candidate)))
        for k in range(len(mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(limbSeq[k]) - 1
    
                for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1
    
                    if found == 1:
                        j = subset_idx[0]
                        if (subset[j][indexB] != partBs[i]):
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
    
                    # if find no partA in the subset, create a new subset
                    elif not found and k < numparts:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + \
                                  connection_all[k][i][2]
                        subset = np.vstack([subset, row])
        # delete some rows of subset which has few parts occur
        # deleteIdx = [];
        # for i in range(len(subset)):
        #     if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
        #         deleteIdx.append(i)
        # subset = np.delete(subset, deleteIdx, axis=0)
        temp_parts ={}
        parts={}
        for i in range(numparts):#17
            temp_parts[i]=[]
            for j in range(len(all_peaks[i])):
                a=all_peaks[i][j][0]*resize
                b=all_peaks[i][j][1]*resize
                c=all_peaks[i][j][2]
                temp_parts[i].append([a,b,c])
            parts[i]=temp_parts[i]
        mappings=[]
        for i in range(numparts):#17
            for n in range(len(subset)):
                kind=limbSeq[i]
                index = subset[n][np.array(kind) - 1]
                if -1 in index:
                    continue
                Y = candidate[index.astype(int), 0]
                X = candidate[index.astype(int), 1]
                S = candidate[index.astype(int), 2]
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0]*resize - X[1]*resize) ** 2 + (Y[0]*resize - Y[1]*resize) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0]*resize - X[1]*resize, Y[0]*resize - Y[1]*resize))
                mappings.append([[int(Y[0])*resize,int(X[0])*resize],[int(Y[1])*resize,int(X[1])*resize],np.array(S).mean(),length,angle,kind])
        frame_detections[idx+init]['mapping']=mappings
        frame_detections[idx+init]['parts']=parts
        if show:
            canvas = ori[idx]  # B,G,R order
            for i in range(numparts):#17
                for j in range(len(all_peaks[i])):
                    cv2.circle(canvas, all_peaks[i][j][0:2], 1, colors[i], thickness=-1)
    
            stickwidth = 10//(resize-1) #4
    
            for i in range(numparts):#17
                for n in range(len(subset)):
                    index = subset[n][np.array(limbSeq[i]) - 1]
                    if -1 in index:
                        continue
                    cur_canvas = canvas.copy()
                    Y = candidate[index.astype(int), 0]
                    X = candidate[index.astype(int), 1]
                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                    angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                    polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0,
                                               360, 1)
                    cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
                    canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
            canvas_out.append(canvas)
       
    return frame_detections