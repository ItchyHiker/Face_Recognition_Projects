import os, sys
sys.path.append('.')

import cv2
import numpy as np
import torch
import torchvision.transforms as tfm

from nets.se_resnet import ResNet18
from mtcnn.detector import MtcnnDetector
from align_faces import warp_and_crop_face, get_reference_facial_points


transforms = tfm.Compose([
    tfm.ToTensor(),
    tfm.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def load_face_gallery():
    embeddings = torch.load('./gallery/face_embeddings.pth')
    names = np.load('./gallery/face_names.npy')
    return embeddings, names

if __name__ == '__main__':
    resize_scale = 4
    cap = cv2.VideoCapture('resources/TheBigBangTheory.mp4')
    face_detector = MtcnnDetector()
    face_embedder = ResNet18()
    embeddings, names = load_face_gallery()
    reference_5pts = get_reference_facial_points(
            (112, 112), (0.25, 0.25), (0, 0), True)
    while True:
        _, frame = cap.read()
        h, w, c = frame.shape
        frame = cv2.resize(frame, (w//resize_scale, h//resize_scale))
        bboxes, landmarks = face_detector.detect_faces(frame, thresholds=[0.9, 0.9, 0.9])
        
        if len(bboxes) == 0:
            continue
        
        input_tensors = None
        for i, bbox in enumerate(bboxes):
            landmark = landmarks[i]
            bbox = [int(_) for _ in bbox]
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2, 16)
        
            facial_5pts = np.reshape(landmark, (2, 5))
            aligned_image = warp_and_crop_face(frame, facial_5pts, 
                    reference_pts=reference_5pts, crop_size=(112, 112))
            aligned_image = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)
            input_tensor = transforms(aligned_image).unsqueeze(0)
            if input_tensors is None:
                input_tensors = input_tensor
            else:
                input_tensors = torch.cat((input_tensors, input_tensor), 0)
        print(input_tensors.shape)


        with torch.no_grad():
            embeddings = face_embedder(input_tensors)
        print(embeddings.shape)


        cv2.imshow("frame", frame)
        cv2.waitKey(1)
