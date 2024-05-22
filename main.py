import os
from face_sdk_3divi.modules import processing_block
from face_sdk_3divi import Service

from file_hndler import get_image_files


sdk_path = 'C:\Users\Mensh\3DiVi_FaceSDK\3_22_0'
service = Service.create_service(sdk_path)

configCtx = {"unit_type": "QUALITY_ASSESSMENT_ESTIMATOR", "modification": "assessment"}
blockQuality = service.create_processing_block(configCtx)

 imgCtx = {
    "image" : {
    "format": "NDARRAY",
        "blob": "data pointer",
        "dtype": "uint8_t",
        "shape": [height, width, channels]
    },
    "objects": [{
        "id": {"type": "long", "minimum": 0},
        "class": "face",
        "confidence": {"double",  "minimum": 0,  "maximum": 1},
        "bbox": [x1, y2, x2, y2]
        "keypoints": {
            "left_eye_brow_left":   {"proj" : [x, y]},
            "left_eye_brow_up":     {"proj" : [x, y]},
            "left_eye_brow_right":  {"proj" : [x, y]},
            "right_eye_brow_left":  {"proj" : [x, y]},
            "right_eye_brow_up":    {"proj" : [x, y]},
            "right_eye_brow_right": {"proj" : [x, y]},
            "left_eye_left":        {"proj" : [x, y]},
            "left_eye":             {"proj" : [x, y]},
            "left_eye_right":       {"proj" : [x, y]},
            "right_eye_left":       {"proj" : [x, y]},
            "right_eye":            {"proj" : [x, y]},
            "right_eye_right":      {"proj" : [x, y]},
            "left_ear_bottom":      {"proj" : [x, y]},
            "nose_left":            {"proj" : [x, y]},
            "nose":                 {"proj" : [x, y]},
            "nose_right":           {"proj" : [x, y]},
            "right_ear_bottom":     {"proj" : [x, y]},
            "mouth_left":           {"proj" : [x, y]},
            "mouth":                {"proj" : [x, y]},
            "mouth_right":          {"proj" : [x, y]},
            "chin":                 {"proj" : [x, y]},
            "points": ["proj": [x, y]]
        }
    }]
}

files_path = 'C:\Users\Mensh\Desktop\lfw\lfw'
image_files = get_image_files(files_path)
for image in image_files:
    print(image)


def qulity_scan(sdk_path, file_path):
    if not os.path.exists(file_path):
        raise Exception(f"not exist file {file_path}")
    