import click,glob
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_folder', type=click.Path(exists=True))
@click.argument('output_folder', type=click.Path())
@click.argument('video_folder',default='/mnt/storage/Gurabo/videos/Gurabo/mp4/',type = click.Path(exist=True))
def extract_images_dataset(input_folder,output_folder,video_folder):
    """
    Place your annotations file from labelbee interface in a folder. This script will take all the .json files, look for the videos with the same name in the given video_folder and extract the images regarding your dataset. 
    """
    
    logger = logging.getLogger(__name__)
    
    annot_files = glob.glob(os.path.join(input_folder,'*.json'))
    counter=0
    for file in annot_files:
        logger.info(file)
        label_annotations = load(file)
        annot= label_annotations['data']
        core = file.split('/')[-1].split('-')[0]
        path_video = os.path.join(video_folder,'%s.mp4'%core)
        video = cv2.VideoCapture(path_video)
        output_folder_training = os.path.join(output_folder,'bee_pose_raw')
        #output_folder_testing = os.path.join(output_folder,'testing')
        os.makedirs(output_folder,exist_ok=True)
        os.makedirs(output_folder_training,exist_ok=True)
        #os.makedirs(output_folder_testing,exist_ok=True)
        training_flag = 100
        not_extracted = []
        for f,frame in enumerate(annot.keys()):
            counter +=1
       
            video.set(cv2.CAP_PROP_POS_FRAMES,int(frame))
            t,im=video.read()
            if t == False:
                logger.warning('%s not extracted'%frame)
                not_extracted.append(frame)
                continue 
            #if f < training_flag+1:
            file_name = os.path.join(output_folder_training,'%012d.jpg'%int(counter))
            cv2.imwrite(file_name,im)
            #else: 
            #    file_name = os.path.join(output_folder_testing,'%012d.jpg'%int(counter))
            #    cv2.imwrite(file_name,im)
        
        
    
    

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
