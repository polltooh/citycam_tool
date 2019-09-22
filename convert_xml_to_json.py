from __future__ import division

import json
import os
import xml.etree.ElementTree as ET
import xmltodict

# import cv2
import numpy as np


def meta_property():
    classes = ['vehicle']
    meta = {
        'dataset_name': 'WebCamT',
        'image_width': 352,
        'image_height': 240,
        'classes': classes,
        'num_classes': 11
    }
    return meta


# def draw_boxes(image, boxes, c, thickness):
#     box_num = boxes.shape[0]
#     for i in range(box_num):
#         cv2.rectangle(image, (int(boxes[i][0]), int(boxes[i][1])),
#                       (int(boxes[i][2]), int(boxes[i][3])), c, thickness)
#     return image


def order_dict_2_box(order_dict):
    bbox = order_dict['bndbox']
    box = [bbox['ymin'], bbox['xmin'], bbox['ymax'], bbox['xmax']]
    box = [float(b) for b in box]
    return box


def parse_annot(annot_name):
    assert(annot_name.endswith(".xml"))
    with open(annot_name) as xml_d:
        ss = xml_d.read()
        try:
            doc = xmltodict.parse(ss)
        except:
            try:
                ss = ss.replace("&", "")
                doc = xmltodict.parse(ss)
            except:
                print(annot_name + " cannot be read")

    bbox_list = list()
    type_list = list()
    if 'vehicle' not in doc['annotation']:
        print(annot_name + " no vehicle")
    else:
        if isinstance(doc['annotation']['vehicle'], list):
            for vehicle in doc['annotation']['vehicle']:
                box = order_dict_2_box(vehicle)
                vehicle_type = int(vehicle['type'])
                bbox_list.append(box)
                type_list.append(vehicle_type)
        else:
            vehicle = doc['annotation']['vehicle']
            vehicle_type = int(vehicle['type'])
            bbox_list = [order_dict_2_box(vehicle)]
            type_list.append(vehicle_type)

    image_name = annot_name.replace('.xml', '.jpg')
    assert os.path.exists(image_name)
    mask_name = '/'.join(image_name.split('/')[:-1]) + '_msk.png'
    assert os.path.exists(mask_name)

    file_dict = {}
    file_dict['labels'] = type_list
    file_dict['bboxes'] = bbox_list
    file_dict['image_name'] = image_name.encode('utf-8').replace(data_dir, "")
    file_dict['mask_name'] = mask_name.encode('utf-8').replace(data_dir, "")

    return file_dict


def full_path_listdir(data_dir):
    return [os.path.join(data_dir, f) for f in os.listdir(data_dir)]


def txt_to_json(data_dir):
    print('Processing {}'.format(data_dir))
    json_list = []
    cam_list = [f for f in full_path_listdir(data_dir) if os.path.isdir(f)]
    for cam in cam_list:
        full_path = cam
        annotation_list = [os.path.join(full_path, f) for f in os.listdir(
            full_path) if f.endswith('xml')]
        for annot in annotation_list:
            file_dict = parse_annot(annot)
            if file_dict != {}:
                json_list.append(file_dict)

    return json_list


def json_list_for_single_cam(data_dir, cam_num):
    json_list = txt_to_json(os.path.join(data_dir, cam_num))

    with open('{}.json'.format(cam_num), 'w') as f:
        json.dump(json_list, f, indent=4)

    max_num_box = max([len(file_dict['bboxes']) for file_dict in json_list])
    return max_num_box

if __name__ == "__main__":
    meta = meta_property()
    meta_json = {'meta': meta}
    cam_num = '164'
    data_dir = '../citycam_dataset/'
    max_num_box = json_list_for_single_cam(data_dir, cam_num)

    meta_json['meta']['max_num_box'] = max_num_box

    with open('WebCamT_meta.json', 'w') as f:
        json.dump(meta_json, f, indent=4)
