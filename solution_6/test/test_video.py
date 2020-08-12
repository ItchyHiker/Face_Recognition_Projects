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

def normalize_feature(face_embeddings):
    norm_face_embeddings = face_embeddings 
    norm = torch.sqrt(torch.sum(torch.pow(norm_face_embeddings, 2), 1, keepdim=True))
    norm_face_embeddings = norm_face_embeddings / norm
    return norm_face_embeddings

if __name__ == '__main__':
    resize_scale = 1
    cap = cv2.VideoCapture('resources/TheBigBangTheory.mp4')
    face_detector = MtcnnDetector()
    face_embedder = ResNet18()
    target_embeddings, names = load_face_gallery()
    norm_target_embeddings = normalize_feature(target_embeddings)
    
    reference_5pts = get_reference_facial_points(
            (112, 112), (0.25, 0.25), (0, 0), True)
    
    _, frame = cap.read()

    video_out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 
            20.0, (frame.shape[1], frame.shape[0]), isColor=True)
    
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

        with torch.no_grad():
            source_embeddings = face_embedder(input_tensors)
        
        norm_source_embeddings = normalize_feature(source_embeddings)
        # calculate distance between detected face embedding feature and feature in gallery
        # cosine score
        scores = torch.sum(norm_target_embeddings*norm_source_embeddings, 1)
        max_score, max_idx = torch.max(scores, dim=0)
        max_score = max_score.cpu().numpy()
        max_idx = max_idx.cpu().numpy()
        if max_score > 0.2568:
            print("Howard in this image")
            bbox = bboxes[max_idx]
            bbox = [int(_) for _ in bbox] 
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2, 16)
            cv2.putText(frame, "Howard", (bbox[0]-20, bbox[1]-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        else:
            print("Howard not in this image")
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
        video_out.write(frame)

    cap.release()
    video_out.release()
    cv2.destroyAllWindows()
