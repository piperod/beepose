import os,glob
import argparse

def avi2mp4(file,output):
    os.system('ffmpeg -loglevel error -y -i "%s" -movflags faststart -an -vcodec copy "%s"'%(str(file),output))
    
def avi2mp4_folder(input_folder,output_folder=''):
    
    if output_folder=='':
        output_folder = folder
        
    os.makedirs(output_folder,exist_ok=True)
    files = glob.glob(os.path.join(folder,'*.avi'))
    processed_files=  glob.glob(os.path.join(folder,'*.mp4'))
   
    for f in files:
        print('processing file', f,'...output:',f.split('/')[-1][:-3]+"mp4")
        output = os.path.join(output_folder,f.split('/')[-1][:-3]+'mp4')
        if output in processed_files:
            print('File %s Already processed skipping'%f)
            continue 
        avi2mp4(f,output)
    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default ='detections/one_week/', help='input folder of videos avi')
    parser.add_argument('--output_folder', type=str, default ='', help='output folder of videos mp4')
    
    args = parser.parse_args()
    folder = args.input_folder
    output_folder = args.output_folder
    
    
       