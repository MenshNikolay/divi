import os

def get_image_files(files_path):
    image_extensions = ('.png', '.bmp', '.tif', '.tiff', '.jpg', '.jpeg', '.ppm')
    image_files = []
    for root, _, files in os.walk(files_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, file))
    return image_files