import json 
import argparse
import glob,os
from beepose.utils.util import read_json

def merging(filenames):
    
    """
    This function takes a list of filenames to be merged. There has to be at least two files. And they should be given in order. 
    The final name will be the name of the first video with the prefix "merged".
    
    Input: List of filenames to merge
    """
    
    assert len(filenames)>1, "Not enough files to merge"
    
    det_1=read_json(filenames[0])
    det_2=read_json(filenames[1])
    print('reading done')
    merged = dict(det_1,**det_2)
    print('first file merged')
    if len(filenames)==2:
        print('Only two files merged.')
        return merged
    
    for n in range(2,len(filenames)):
        det_3= read_json(filenames[n])
        merged.update(det_3)
        print('%d file merged'%n)
    
    prefix = '/'.join(filenames[0].split('/')[:-1])+'/'
    filename = filenames[0].split('/')[-1]
    output_name= prefix+'merged_'+filename[1:]
    with open(output_name, 'w') as outfile:
        json.dump(merged, outfile)
    print('Success')
    return output_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_name', type=str, default ='detections/', help='input folder')
    parser.add_argument('--num_files', type=int, default =3, help='number of files to merge')
    args = parser.parse_args()
    folder_name=args.folder_name
    num=args.num_files
    detfiles = glob.glob(os.path.join(folder_name,'1*.json'))
    for l in detfiles:
        print(l)
        prefix ='/'.join(l.split('/')[:-1])+'/'
        video_name =l.split('/')[-1][1:]
        print(video_name)
        listfiles=[prefix+str(i)+video_name for i in range(1,num+1)]
        print(listfiles)
        merging(listfiles)