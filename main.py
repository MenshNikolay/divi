import argparse 
import time
from tqdm import tqdm

from file_handler import get_image_files
from qaa import quality_estimator
from sri import select_random_images
from plot import plot_histogram




def main(directory,num_processed=None):
    images = get_image_files(directory)
    if num_processed is None:
        selected_img = images
        print(selected_img)
    else:
        selected_img = select_random_images(images=images, num=num_processed)
        print(selected_img)
    for path in tqdm(selected_img):
        quality_estimator(input_image=path, 
                          sdk_path=r"C:\Users\Mensh\3DiVi_FaceSDK\3_22_0", 
                          modification="assessment")
      
        
        

if __name__ == "__main__":
    directory = "C:/Users/Mensh/Desktop/lfw"
    main(directory=directory, num_processed=10)
    plot_histogram(r'C:\Users\Mensh\Desktop\divi\result.csv')