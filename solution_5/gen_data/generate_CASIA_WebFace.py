import os
import argparse

parser = argparse.ArgumentParser(description='Script for generating annotation file')
parser.add_argument('--image_path', type=str, default='training_data/CASIA', 
        help='image data path')
parser.add_argument('--anno_file', type=str, default='annotations/CASIA_anno.txt',
        help='annotation file')
args = parser.parse_args()

image_path = args.image_path
anno_file = args.anno_file
persons = sorted([f for f in os.listdir(image_path) if not f.startswith('.')])

target_file = open(anno_file, 'w')
for i, person in enumerate(persons):
    person_path = os.path.join(image_path, person)
    images = sorted([f for f in os.listdir(person_path) if f.endswith('.jpg')])
    for image in images:
        image_file = os.path.join(person, image)
        target_file.write(image_file+' '+str(i)+'\n')

target_file.close()
