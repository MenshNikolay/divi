import os
import cv2
import numpy as np
import csv
import argparse
from tqdm import tqdm
from sys import platform
from face_sdk_3divi import FacerecService, Config
from face_sdk_3divi.modules.context import Context

def help_message():
    message = f"\n This program is an example of the Quality Assessment Estimator Processing Block integration \n Usage: " \
              " [--input_dir <directory_path>] " \
              " [--sdk_path <sdk_root_dir>] \n"
    print(message)

def parse_args():
    parser = argparse.ArgumentParser(description='Quality Assessment Estimator for Image Dataset')
    parser.add_argument('--input_dir', default= r'C:\Users\Mensh\Desktop\lfw\lfw', type=str, required=True, help='Path to the directory containing images')
    parser.add_argument('--sdk_path', default= r"C:\Users\Mensh\3DiVi_FaceSDK\3_22_0", type=str, help='Path to the 3DiVi Face SDK')
    parser.add_argument('--modification', default="assessment", type=str, help='Modification for quality assessment')
    return parser.parse_args()

def find_images(directory):
    supported_formats = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".ppm"}
    images = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_formats):
                images.append(os.path.join(root, file))
    return images

def write_csv(results, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "filename", "confidence", "totalScore", "isSharp", "sharpnessScore", "isEvenlyIlluminated", 
            "illuminationScore", "noFlare", "isLeftEyeOpened", "isRightEyeOpened", "isRotationAcceptable", 
            "notMasked", "isNeutralEmotion", "isEyesDistanceAcceptable", "eyesDistance", 
            "isMarginsAcceptable", "isNotNoisy", "hasWatermark", "dynamicRangeScore", "isDynamicRangeAcceptable"
        ])
        for result in results:
            writer.writerow([result[field] for field in writer.fieldnames])

def draw_bbox(rect, img, color=(0, 255, 0)):
    return cv2.rectangle(img, (int(rect[0].get_value() * img.shape[1]), int(rect[1].get_value() * img.shape[0])),
                         (int(rect[2].get_value() * img.shape[1]), int(rect[3].get_value() * img.shape[0])), color, 2)

def print_bbox_coordinates(bbox: Context, image):
    x1 = int(bbox[0].get_value() * image.shape[1])
    y1 = int(bbox[1].get_value() * image.shape[0])
    x2 = int(bbox[2].get_value() * image.shape[1])
    y2 = int(bbox[3].get_value() * image.shape[0])
    print(f"Bbox coordinates: ({x1}, {y1}) ({x2}, {y2})")

def quality_estimator(image_path, service, quality_block, capturer):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, im_png = cv2.imencode('.png', image_rgb)
    img_bytes = im_png.tobytes()

    samples = capturer.capture(img_bytes)
    if not samples:
        return None

    image_ctx = {
        "blob": image_rgb.tobytes(),
        "dtype": "uint8_t",
        "format": "NDARRAY",
        "shape": list(image_rgb.shape)
    }

    ioData = service.create_context({"image": image_ctx})
    ioData["objects"] = [sample.to_context() for sample in samples]

    quality_block(ioData)

    results = []
    for obj in ioData["objects"]:
        quality = obj["quality"]
        results.append({
            "filename": os.path.basename(image_path),
            "confidence": quality["confidence"].get_value(),
            "totalScore": quality["total_score"].get_value(),
            "isSharp": quality["is_sharp"].get_value(),
            "sharpnessScore": quality["sharpness_score"].get_value(),
            "isEvenlyIlluminated": quality["is_evenly_illuminated"].get_value(),
            "illuminationScore": quality["illumination_score"].get_value(),
            "noFlare": quality["no_flare"].get_value(),
            "isLeftEyeOpened": quality["is_left_eye_opened"].get_value(),
            "isRightEyeOpened": quality["is_right_eye_opened"].get_value(),
            "isRotationAcceptable": quality["is_rotation_acceptable"].get_value(),
            "notMasked": quality["not_masked"].get_value(),
            "isNeutralEmotion": quality["is_neutral_emotion"].get_value(),
            "isEyesDistanceAcceptable": quality["is_eyes_distance_acceptable"].get_value(),
            "eyesDistance": quality["eyes_distance"].get_value(),
            "isMarginsAcceptable": quality["is_margins_acceptable"].get_value(),
            "isNotNoisy": quality["is_not_noisy"].get_value(),
            "hasWatermark": quality["has_watermark"].get_value(),
            "dynamicRangeScore": quality["dynamic_range_score"].get_value(),
            "isDynamicRangeAcceptable": quality["is_dynamic_range_acceptable"].get_value()
        })
    return results

def main(directory, sdk_path, modification):
    sdk_conf_dir = os.path.join(sdk_path, 'conf', 'facerec')
    if platform == "win32":
        sdk_dll_path = os.path.join(sdk_path, 'bin', 'facerec.dll')
        sdk_onnx_path = os.path.join(sdk_path, 'bin')
    else:
        sdk_dll_path = os.path.join(sdk_path, 'lib', 'libfacerec.so')
        sdk_onnx_path = os.path.join(sdk_path, 'lib')

    service = FacerecService.create_service(sdk_dll_path, sdk_conf_dir, f'{sdk_path}/license')

    quality_config = {
        "unit_type": "QUALITY_ASSESSMENT_ESTIMATOR",
        "modification": modification,
        "ONNXRuntime": {"library_path": sdk_onnx_path}
    }
    if modification == "assessment":
        quality_config["config_name"] = "quality_assessment.xml"

    quality_block = service.create_processing_block(quality_config)

    capturer_config = Config("common_capturer_uld_fda.xml")
    capturer = service.create_capturer(capturer_config)

    images = find_images(directory)

    results = []
    for image_path in tqdm(images, desc="Processing images"):
        result = quality_estimator(image_path, service, quality_block, capturer)
        if result:
            results.extend(result)

    write_csv(results, "result.csv")

if __name__ == "__main__":
    help_message()
    args = parse_args()
    main(args.input_dir, args.sdk_path, args.modification)
