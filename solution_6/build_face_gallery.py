import os, sys

import cv2
import numpy as np
import torch
import torchvision.transforms as tfs

from nets.se_resnet import ResNet18
from mtcnn.detector import MtcnnDetector
from align_faces import warp_and_crop_face, get_reference_facial_points

transforms = tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def register_face(detect_model, embedding_model, image_path):
    embedding_model.eval()

    embeddings = []
    names = []
    
    images = [f for f in os.listdir(image_path) if f.endswith('.jpg')]
    for image_name in images:
        
        person_name = image_name.split('.')[0]

        image = cv2.imread(os.path.join(image_path, image_name), 1)
        
        # 1. detect face bboxes and landmarks from image
        bboxes, landmarks = detect_model.detect_faces(image)
        assert len(bboxes) == 1, "There must be only one face in this image"
        
        # 2. face alignment with landmarks
        facial_5points = np.reshape(landmarks, (2, 5))
        reference_5pts = get_reference_facial_points(
                (112, 112), (0.25, 0.25), (0, 0), True)
        aligned_image = warp_and_crop_face(image, facial_5points, reference_pts=reference_5pts, crop_size=(112, 112))
        cv2.imshow("aligned_image", aligned_image)
        cv2.waitKey(0)
        # 3. face embedding using face recognition model
        aligned_image = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)
        input_tensor = transforms(aligned_image).unsqueeze(0)
        with torch.no_grad():
            embedding_feature = embedding_model(input_tensor)
            embeddings.append(embedding_feature)

        # 4. register face into face gallery
        print("Register face:", person_name)
        names.append(person_name)

if __name__ == "__main__":
    detect_model = MtcnnDetector()
    embedding_model = ResNet18()
    image_path = './images'

    register_face(detect_model, embedding_model, image_path)
