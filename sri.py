import random 


def select_random_images(images, num):
    if num and num < len(images):
        return random.sample(images, num)
    
    