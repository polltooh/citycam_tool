from __future__ import division

import json
import os
import xml.etree.ElementTree as ET
import xmltodict

# import cv2
import numpy as np


def meta_property():
    label_map = {
        0: 'Unknown',
        1: 'Taxi',
        2: 'Black sedan',
        3: 'Other car',
        4: 'Little truck',
        5: 'Middle truck',
        6: 'Big truck',
        7: 'Van',
        8: 'Middle bus',
        9: 'Big bus',
        10: 'Bicyle',
    },
    meta = {
        'dataset_name': 'WebCamT',
        'image_width': 352,
        'image_height': 240,
        'label_map': label_map,
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


def parse_annot(annot_name, data_dir):
    # Remove camera number in in the end.
    data_dir = os.path.dirname(data_dir)
    # Adds '/' in the end.
    data_dir += '/'

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
    if not os.path.exists(image_name):
        print(image_name + ' does not exist.')
        return {}

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


def txt_to_json(data_dir, select_set):
    print('Processing {}'.format(data_dir))
    json_list = []
    cam_path_list = [f for f in full_path_listdir(data_dir) if os.path.isdir(f)]
    for cam_path in cam_path_list:
        cam_basename = os.path.basename(cam_path)
        if cam_basename not in select_set:
            continue

        annotation_list = [os.path.join(cam_path, f) for f in os.listdir(
            cam_path) if f.endswith('xml')]
        for annot in annotation_list:
            file_dict = parse_annot(annot, data_dir)
            if file_dict != {}:
                json_list.append(file_dict)

    return json_list


def json_list_for_single_cam(data_dir, cam_num, select_set):
    json_list = txt_to_json(os.path.join(data_dir, cam_num), select_set)

    # with open('{}.json'.format(cam_num), 'w') as f:
    #     json.dump(json_list, f, indent=4)

    # max_num_box = max([len(file_dict['bboxes']) for file_dict in json_list])
    # return max_num_box
    return json_list


def read_txt_file(file_path):
    with open(file_path, 'r') as f:
        file_set = set(f.read().replace('\n','').split('\r'))
    return file_set


def read_train_test_separation(data_dir):
    downtown_train = read_txt_file(os.path.join(
        data_dir, 'train_test_separation/Downtown_Train.txt'))
    downtown_test = read_txt_file(os.path.join(
        data_dir, 'train_test_separation/Downtown_Test.txt'))
    parkway_train = read_txt_file(os.path.join(
        data_dir, 'train_test_separation/Parkway_Train.txt'))
    parkway_test = read_txt_file(os.path.join(
        data_dir, 'train_test_separation/Parkway_Test.txt'))
    return downtown_train, downtown_test, parkway_train, parkway_test


def generate_json(data_dir, cam_list, select_set, save_path):
    json_list = []
    for cam in cam_list:
        json_list.extend(json_list_for_single_cam(data_dir, cam, select_set))

    with open(save_path, 'w') as f:
        json.dump(json_list, f, indent=4)

def main():
    meta = meta_property()
    meta_json = {'meta': meta}
    # data_dir = '../traffic_video_analysis/data/CityCam/'
    data_dir = '../citycam_dataset/CityCam/'
    cam_list = ['164','166','170','173','181','253','398','403','410','495','511','551','572','691','846','928','bigbus']
    downtown_train, downtown_test, parkway_train, parkway_test = read_train_test_separation(data_dir)
    generate_json(data_dir, cam_list, downtown_train, 'Downtown_Train.json')
    generate_json(data_dir, cam_list, downtown_test, 'Downtown_Test.json')
    generate_json(data_dir, cam_list, parkway_train, 'Parkway_Train.json')
    generate_json(data_dir, cam_list, parkway_test, 'Parkway_Test.json')

    # json_list = json_list_for_single_cam(data_dir, cam_list[0], downtown_train)

    # meta_json['meta']['max_num_box'] = max_num_box

    with open('WebCamT_meta.json', 'w') as f:
        json.dump(meta_json, f, indent=4)


if __name__ == "__main__":
    main()
