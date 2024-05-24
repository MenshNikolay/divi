import argparse 
import time
from tqdm import tqdm

from file_handler import get_image_files
from qaa import quality_estimator
from sri import select_random_images

directory = "C:/Users/Mensh/Desktop/lfw"


def main(directory,num_processed):
    images = get_image_files(directory)
    selected_img = select_random_images(images=images, num=num_processed)
    for path in tqdm(selected_img):
        quality_estimator(input_image=path, 
                          sdk_path=r"C:\Users\Mensh\3DiVi_FaceSDK\3_22_0", 
                          modification="assessment")
      
        
        

if __name__ == "__main__":
    
    main(directory="C:/Users/Mensh/Desktop/lfw", num_processed=15)