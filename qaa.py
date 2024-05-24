import argparse  # for an ArgumentParser
import os.path
import cv2  # for reading images, visualising window and rectangle creation
import numpy as np


from sys import platform  # for a platform identification
from face_sdk_3divi import FacerecService, Config  # FacerecService creates service, Config creates capturer config
from face_sdk_3divi.modules.context import Context

from write_csv import save_to_csv


def help_message():
    message = f"\n This demo uses QAE Processing Block integration with assessment mod read more(https://docs.3divi.ai/) \n Usage: " \
              " [--input_image <image_path>] " \
              " [--num_processed <number_of_images>] "\
              " [--use_cuda <--use_cuda>] " \
              " [--sdk_path <sdk_root_dir>] \n"
    print(message)


def parse_args():  # launch parameters
    parser = argparse.ArgumentParser(description='Processing Block Example')
    parser.add_argument('--input_image', type=str, required=True)
    parser.add_argument('--modification', default="assessment", type=str)
    parser.add_argument('--sdk_path', default=r"C:\Users\Mensh\3DiVi_FaceSDK\3_22_0", type=str)
    parser.add_argument('--num_processed', type=int, help='Number of images to process')
    parser.add_argument('--use_cuda', action='store_true')  # pass the '--use_cuda' parameter before launch to use cuda acceleration
    return parser.parse_args()

'''при пакетной обработке не нужно
def draw_bbox(rect, img, color=(0, 255, 0)):  # an example of a bbox drawing with opencv
    return cv2.rectangle(img, (int(rect[0].get_value() * img.shape[1]), int(rect[1].get_value() * img.shape[0])),
                         (int(rect[2].get_value() * img.shape[1]), int(rect[3].get_value() * img.shape[0])), color, 2)


def print_bbox_coordinates(bbox: Context, image):
    x1 = int(bbox[0].get_value() * image.shape[1])
    y1 = int(bbox[1].get_value() * image.shape[0])
    x2 = int(bbox[2].get_value() * image.shape[1])
    y2 = int(bbox[3].get_value() * image.shape[0])
    print(
        f"Bbox coordinates: ({x1}, {y1}) ({x2}, {y2})")
'''

def quality_estimator(input_image, sdk_path, modification):
    sdk_conf_dir = os.path.join(sdk_path, 'conf', 'facerec')
    if platform == "win32":  # for Windows
        sdk_dll_path = os.path.join(sdk_path, 'bin', 'facerec.dll')
        sdk_onnx_path = os.path.join(sdk_path, 'bin')
    else:  # for Linux
        sdk_dll_path = os.path.join(sdk_path, 'lib', 'libfacerec.so')
        sdk_onnx_path = os.path.join(sdk_path, 'lib')

    service = FacerecService.create_service(  # create FacerecService
        sdk_dll_path,
        sdk_conf_dir,
        f'{sdk_path}/license')

    quality_config = {  # quality block configuration parameters
        "unit_type": "QUALITY_ASSESSMENT_ESTIMATOR",  # required parameter
        "modification": modification,
        "ONNXRuntime": {
            "library_path": sdk_onnx_path  # optional
        }
    }
    if (modification == "assessment"):
        quality_config["config_name"] = "quality_assessment.xml"

    quality_block = service.create_processing_block(
        quality_config)  # create quality assessment estimation processing block

    capturer_config = Config("common_capturer_uld_fda.xml")
    capturer = service.create_capturer(capturer_config)

    img: np.ndarray = cv2.imread(input_image, cv2.IMREAD_COLOR)  # read an image from a file
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert an image in RGB for correct results
    _, im_png = cv2.imencode('.png', image)  # image encoding, required for convertation in raw sample
    img_bytes = im_png.tobytes()  # copy an image to a byte string

    samples = capturer.capture(img_bytes)  # capture faces in an image

    image_ctx = {  # put an image in container
        "blob": image.tobytes(),
        "dtype": "uint8_t",
        "format": "NDARRAY",
        "shape": [dim for dim in image.shape]
    }

    ioData = service.create_context({"image": image_ctx})
    ioData["objects"] = []
    results = []

    for sample in samples:  # iteration over detected faces in ioData container
        ctx = sample.to_context()
        ioData["objects"].push_back(ctx)  # add results to ioData container

    quality_block(ioData)  # call an estimator and pass a container with a cropped image
    '''В данном блоке реализуется функционал по подготовке данных для записи в csv фай.
       Применяется метод преобразования в стандартный тип данных для python, словарь.
       нужно добавить название файла. название будет соответсвовать обсалютному пути файла,
       это упростит его поиск при использовании большого датасета.
    '''
    io_dic = ioData.to_dict()
    
    if io_dic is None:
        print(f"Error: io_dic is None for input image {input_image}")
        return
    

    for obj in io_dic["objects"]:# iteration over objects in io_dic
        quality_params = obj["quality"]
        meta_data = {
            "filename": input_image,
            "confidence": obj["confidence"],
            "total_score": quality_params["total_score"],
            "is_sharp": quality_params["is_sharp"],
            "background_uniformity_score": quality_params["background_uniformity_score"],
            "dynamic_range_score": quality_params["dynamic_range_score"],
            "eyes_distance": quality_params["eyes_distance"],
            "has_watermark": quality_params["has_watermark"],
            "illumination_score": quality_params["illumination_score"],
            "is_background_uniform": quality_params["is_background_uniform"],
            "is_dynamic_range_acceptable": quality_params["is_dynamic_range_acceptable"],
            "is_evenly_illuminated": quality_params["is_evenly_illuminated"],
            "is_eyes_distance_acceptable": quality_params["is_eyes_distance_acceptable"],
            "is_left_eye_opened": quality_params["is_left_eye_opened"],
            "is_margins_acceptable":quality_params["is_margins_acceptable"],
            "is_neutral_emotion": quality_params["is_neutral_emotion"],
            "is_not_noisy": quality_params["is_not_noisy"],
            "is_right_eye_opened": quality_params["is_right_eye_opened"],
            "is_rotation_acceptable": quality_params["is_rotation_acceptable"],
            "left_eye_openness_score": quality_params["left_eye_openness_score"],
            "margin_inner_deviation": quality_params["margin_inner_deviation"],
            "margin_outer_deviation": quality_params["margin_outer_deviation"],
            "max_rotation_deviation": quality_params["max_rotation_deviation"],
            "neutral_emotion_score": quality_params["neutral_emotion_score"],
            "no_flare": quality_params["no_flare"],
            "noise_score": quality_params["noise_score"],
            "not_masked": quality_params["not_masked"],
            "not_masked_score": quality_params["not_masked_score"],
            "right_eye_openness_score": quality_params["right_eye_openness_score"],
            "sharpness_score": quality_params["sharpness_score"],
            "watermark_score": quality_params["watermark_score"]
        }
        results.append(meta_data)
    save_to_csv(results, "result.csv")
        
    '''
    Не требуется для пакетной обработки данных
    cv2.imshow("result", picture)  # an example of a result image visualizing with opencv
    cv2.waitKey(0)  # wait for a key to be pressed to close the window
    cv2.destroyAllWindows()  # close the window
    '''




if __name__ == "__main__":
    help_message()
    args = parse_args()

    quality_estimator(args.input_image, args.sdk_path, args.modification)
    
    