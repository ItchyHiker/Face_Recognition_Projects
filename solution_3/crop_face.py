import os, sys

from mtcnn.detector import MtcnnDetector
from deepblue_align_mtcnn import process

casia_data_path = './data/CASIA-WebFace/'
lfw_data_path = './data/lfw'

casia_persons = [f for f in os.listdir(casia_data_path)]
lfw_persons = [f for f in os.listdir(lfw_data_path)]
trainingdata_path = './training_data/'

for person in sorted(casia_person):
    dst_data_path = os.path.join(trainingdata_path, 'CASIA', person)
    if not os.path.exists(dst_data_path):
        os.makedirs(dst_data_path)

    images_path = os.path.join(casia_data_path, person)
    images = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
    for image_file in images:
        process(detector, os.path.join(images_path, image_file), 'unlabelled',
                (112, 112), savepath=dst_data_path)


for person in sorted(LFW_person):
    dst_data_path = os.path.join(trainingdata_path, 'LFW', person)
    if not os.path.exists(dst_data_path):
        os.makedirs(dst_data_path)

    images_path = os.path.join(lfw_data_path, person)
    images = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
    for image_file in images:
        process(detector, os.path.join(images_path, image_file), 'unlabelled',
                (112, 112), savepath=dst_data_path)

