import os
import skimage.io
from app_config.paths import repo_path
from vendor.mrcnn import visualize
import tensorflow as tf
import cv2
import argparse
from silence_tensorflow import silence_tensorflow
from vendor.mrcnn.config import Config as Mask_RCNN_Config
from vendor.mrcnn import model as modellib, utils
from pipelines.runtime_utils import collect_image_paths

silence_tensorflow()
# # 아레에서부터  try catch문까지는 메모리가 적어서 터져버리네요 그래서 제한 했습니다
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#     try:
#         tf.config.experimental.set_virtual_device_configuration(
#             gpus[0],
#             # [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15360)])
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     except RuntimeError as e:
#         print(e)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
ROOT_DIR = os.path.abspath("./")
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
# tf.debugging.set_log_device_placement(True)
import numpy as np

get_images_paths = collect_image_paths

class MaskRCNNSegmenter:
    def __init__(self,mrcnn_config):


        self.config = mrcnn_config
        self.model = modellib.MaskRCNN(mode="inference", config=self.config, model_dir=str(self.config.logs))
        weights_path = self.config.weights
        self.model.load_weights(weights_path, by_name=True)

    def __get_lesion_roi(self, temp_image, rois):
        lis = [None for x in range(len(rois))]
        for i, data in enumerate(rois):
            y1, x1, y2, x2 = data
            temp = temp_image[y1:y2, x1:x2]
            lis[i] = temp
        return lis

    def __set_crop_scale(self, rois, crop_scale, image_shape):
        rows, cols, _ = image_shape
        for i, data in enumerate(rois):
            y1, x1, y2, x2 = data
            center_x = (max(x1, x2) + min(x1, x2)) // 2
            center_y = (max(y1, y2) + min(y1, y2)) // 2
            half_x = max(center_x, x1) - min(center_x, x1)
            half_y = max(center_y, y1) - min(center_y, y1)
            re_y1 = center_y - half_y * crop_scale
            re_x1 = center_x - half_x * crop_scale
            re_y2 = center_y + half_y * crop_scale
            re_x2 = center_x + half_x * crop_scale
            if (re_y1 < 0): re_y1 = 0
            if (re_x1 < 0): re_x1 = 0
            if (re_y2 > rows): re_y2 = rows
            if (re_x2 > cols): re_x2 = cols
            rois[i] = [re_y1, re_x1, re_y2, re_x2]
        return rois
    def inference(self, image_info=None, display=False, save_path=1, show_mask=True, show_bbox=False,
                  show_contour=False, show_label=False, crop_scale = 1):
        image = None
        if isinstance(image_info, str):
            image = skimage.io.imread(image_info)
        elif isinstance(image_info, np.ndarray):
            #  cv2로 이미지를 읽는 상황이라 가정함
            image = cv2.cvtColor(image_info,cv2.COLOR_BGR2RGB)

        assert image is not None, "경로 또는 ndarray를 입력하세요"
        image_shape = image.shape

        if (image.shape[2] == 4):
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        results = self.model.detect([image], verbose=0)
        r = results[0]
        r['rois'] = self.__set_crop_scale(rois=r['rois'], crop_scale=crop_scale,image_shape=image_shape)
        temp_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        roi_list = self.__get_lesion_roi(temp_image, rois=r['rois'])
        confidences = r['scores']
        # print(r['scores'])
        status = True
        N = r['rois'].shape[0]
        # print("here", r['scores'])
        # exit()
        inference_results = []
        if not N:
            status = False
            # inference_results.append((image, [image], status, [0.0]))
            inference_results.append((image, [], status, [0.0]))
        else:
            masked_image, cropped_images, confidences = visualize.display_instances_for_class(image, r['rois'], r['masks'],
                                                                                              r['class_ids'],
                                                                                              self.config.class_names,
                                                                                              r['scores'],
                                                                                              show_mask=show_mask,
                                                                                              show_bbox=show_bbox,
                                                                                              show_contour=show_contour,
                                                                                              show_label=show_label)
            # print(confidences)
            if (display):
                result = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
                for i, data in enumerate(cropped_images):
                    cv2.imshow(str(confidences[i]), cv2.cvtColor(data, cv2.COLOR_RGB2BGR))
                cv2.namedWindow("lesion_segmentation", cv2.WINDOW_NORMAL)
                cv2.imshow("lesion_segmentation", result)
                cv2.waitKey(10)
                cv2.destroyAllWindows()

            inference_results.append((masked_image, cropped_images,status, confidences))
        return inference_results



if __name__ == '__main__':

    class mrcnn_config(Mask_RCNN_Config):
        class_names = ['others', 'atopic_dermatitis', "seborrheic dermatitis", "psoriasis", "rosacea", "acne"]
        weights = repo_path("models", "weights", "segmentation", "mrcnn", "lesion_mask_rcnn.h5")
        NAME = "atopy"
        logs = DEFAULT_LOGS_DIR,
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + 1
        VALIDATION_STEPS = 50
        STEPS_PER_EPOCH = 1000
        DETECTION_MIN_CONFIDENCE = 0.9
        # GPU_COUNT = 1
        # IMAGES_PER_GPU = 1


    mrcnn = MaskRCNNSegmenter(mrcnn_config=mrcnn_config())
    image_path = repo_path("test_data", "images", "samples", "normal_002.JPG")
    asd=mrcnn.inference(image_info=image_path, display=True, show_label=True, show_bbox=False)

    # cv2.imshow("sdf",masked_image)
    # cv2.waitKey(0)

    cv2.imwrite("sdf.jpg",cv2.cvtColor(asd[0][0],cv2.COLOR_RGB2BGR))
    # image_path_list = get_images_paths(image_path)
    # # a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # # b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    # # c = tf.matmul(a, b)
    # #
    # # print(c)
    # # exit()
    # for i, image_path in enumerate(image_path_list):
    #     mrcnn.inference(image_info=image_path, display=True, show_label =True,show_bbox=True)
    #     cv2.waitKey(0)
    #     # break


Mask_RCNN = MaskRCNNSegmenter



