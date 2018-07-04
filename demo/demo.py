import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import cv2

from spn import object_localization
import experiment.util as utils

plt.rcParams["figure.figsize"] = (8,8)

DATA_ROOT = '../data/voc/VOCdevkit/VOC2007'
ground_truth = utils.load_ground_truth_voc(DATA_ROOT, 'trainval')

model_path = './logs/voc2007/model.pth.tar'
model_dict = utils.load_model_voc(model_path)

# predictions = []
# for img_idx in tqdm(range(len(ground_truth['image_list']))):
#     image_name = os.path.join(DATA_ROOT, 'JPEGImages', ground_truth['image_list'][img_idx] + '.jpg')
#     _, input_var = utils.load_image_voc(image_name)
#     gt_labels = (ground_truth['gt_labels'][img_idx] >= 0).nonzero()[0]
#     preds, labels = object_localization(model_dict, input_var, location_type='point', gt_labels=gt_labels, multi_objects=False)
#     predictions += [(img_idx,) + p for p in preds]
#
# print('Pointing accuracy: {:.2f}'.format(utils.pointing(np.array(predictions), ground_truth) * 100.))


def show():
    for img_idx in tqdm(range(len(ground_truth['image_list']))):
        image_name = os.path.join(DATA_ROOT, 'JPEGImages', ground_truth['image_list'][img_idx] + '.jpg')
        _, input_var = utils.load_image_voc(image_name)
        gt_labels = (ground_truth['gt_labels'][img_idx] >= 0).nonzero()[0]
        preds, labels = object_localization(model_dict, input_var, location_type='bbox', gt_labels=gt_labels, multi_objects=False)
        img = Image.open(image_name)
        img_draw = utils.draw_bboxes(img, np.array(preds), ground_truth['class_names'])
        # plt.imshow(img_draw)
        b,g,r = img_draw.split()
        img_draw = Image.merge("RGB",(r,g,b))
        img_array = np.asarray(img_draw)
        cv2.imshow('', img_array)
        cv2.waitKey()

def write():
    tol = 0
    for img_idx in tqdm(range(len(ground_truth['image_list']))):
        tol += 1
        # if tol > 20:
        #     break
        image_name = os.path.join(DATA_ROOT, 'JPEGImages', ground_truth['image_list'][img_idx] + '.jpg')
        _, input_var = utils.load_image_voc(image_name)
        gt_labels = (ground_truth['gt_labels'][img_idx] >= 0).nonzero()[0]
        preds, labels = object_localization(model_dict, input_var, location_type='bbox', gt_labels=gt_labels,
                                            multi_objects=False)
        utils.write_cam(image_name, ground_truth['image_list'][img_idx], np.array(preds), ground_truth['class_names'])


if __name__ == '__main__':
    write()