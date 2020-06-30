import os

import cv2
import numpy as np
from align_faces import warp_and_crop_face, get_reference_facial_points
from mtcnn.detector import MtcnnDetector

def draw_all(filepath, bbox, facial5points, savepath):
    '''Draw face bounding box and landmarks on image.
    '''
    filename = os.path.basename(filepath)
    image = cv2.imread(filepath)
    for i in range(bbox.shape[0]):
        x, y, r, b, _ = list(map(int, bbox[i]))
        lms = list(map(int, facial5points[i]))
        w = r - x + 1
        h = b - y + 1
        cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 2, 16)
        cv2.circle(image, (lms[0], lms[5]), 2, (0, 255, 0), 2)
        cv2.circle(image, (lms[1], lms[6]), 2, (0, 255, 0), 2)
        cv2.circle(image, (lms[2], lms[7]), 2, (0, 255, 0), 2)
        cv2.circle(image, (lms[3], lms[8]), 2, (0, 255, 0), 2)
        cv2.circle(image, (lms[4], lms[9]), 2, (0, 255, 0), 2)
    dst_name = filename[: filename.rfind('.')]
    cv2.imwrite("{}/{}_all.jpg".format(savepath, dst_name), image)


def draw_bbox(filepath, bbox, savepath):
    '''Draw face bounding box on image.
    '''
    image = cv2.imread(filepath)
    filename = os.path.basename(filepath)
    for i in range(bbox.shape[0]):
        x, y, r, b, _ = list(map(int, bbox[i]))
        w = r - x + 1
        h = b - y + 1
        cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 2, 16)
    dst_name = filename[: filename.rfind('.')]
    cv2.imwrite("{}/{}_box.jpg".format(savepath, dst_name), image)

def draw_lms(filepath, facial5points, savepath):
    '''Draw landmarks on image.
    '''
    image = cv2.imread(filepath)
    filename = os.path.basename(filepath)
    for i in range(facial5points.shape[0]):
        lms = list(map(int, facial5points[i]))
        cv2.circle(image, (lms[0], lms[5]), 2, (0, 255, 0), 2)
        cv2.circle(image, (lms[1], lms[6]), 2, (0, 255, 0), 2)
        cv2.circle(image, (lms[2], lms[7]), 2, (0, 255, 0), 2)
        cv2.circle(image, (lms[3], lms[8]), 2, (0, 255, 0), 2)
        cv2.circle(image, (lms[4], lms[9]), 2, (0, 255, 0), 2)
    dst_name = filename[: filename.rfind('.')]
    cv2.imwrite("{}/{}_lms.jpg".format(savepath, dst_name), image)

def draw_name(filepath, bbox, labels, thickness=2):
    '''Draw face identity on image.
    '''
    image = cv2.imread(filepath)
    filename = os.path.basename(filepath)
    x, y, r, b, _ = list(map(int, bbox))
    w = r - x + 1
    h = b - y + 1
    draw = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), thickness, 16)
    pos = (x + 3, y - 5)
    cv2.putText(image, str(labels), pos, 0, 0.5, (0, 255, 0), 2, 16)
    dst_name = filename[: filename.rfind('.')]
    cv2.imwrite("{}/{}_name.jpg".format(savepath, dst_name), draw)

def process(detector, filepath, category, output_size, savepath, visualize=False):
    '''Crop and align face image with MTCNN detector
    Parameters:
    -----------
    detector: face detector which gives face bounding box and 5 landmarks
    filepath: path of image
    category: str of value `labelled` or `unlabelled`
    output_size: tuple of face width and height
    savepath: path for saving the processed image
    '''
    img = cv2.imread(filepath)
    filename = os.path.basename(filepath)
    bbox, facial5points = detector.detect_faces(img)
    
    if visualize:
        # visualization
        draw_bbox(filepath, bbox, savepath)
        draw_lms(filepath, facial5points, savepath)
        draw_all(filepath, bbox, facial5points, savepath)

        # show ref_5pts
        tmp_pts = np.array([[38.29459953, 73.53179932, 56.02519989, 41.54930115, 70.72990036, 51.69630051, 51.50139999,  71.73660278,  92.3655014,
          92.20410156]])
        empty_face = np.zeros((112, 112, 3))
        cv2.imwrite(os.path.join(savepath, 'empty_face.jpg'), empty_face)
        draw_lms(os.path.join(savepath, 'empty_face.jpg'), tmp_pts, savepath)


    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)

    if (len(bbox) > 0):
        if category == 'labelled': # find the max bbox of labelled image
            max_bb = []
            for box in bbox:
                x, y, r, b, _ = list(map(int, box))
                w = r - x + 1
                h = b - y + 1
                max_bb.append(w * h)
            index = max_bb.index(max(max_bb))
            facial5points = facial5points[[index]]
            facial5point = np.reshape(facial5points, (2, 5))
            dst_img = warp_and_crop_face(img, facial5point, reference_pts=reference_5pts, crop_size=output_size)
            dst_name = filename.split('.')[0]
            cv2.imwrite('{}/{}_{}_mtcnn_aligned_{}x{}.jpg'.format(savepath, dst_name, type, output_size[0], output_size[1]), dst_img)
        elif category == 'unlabelled': # process all the detected faces in image
            for i in range(bbox.shape[0]):
                facial5point = np.reshape(facial5points[i], (2, 5))
                dst_img = warp_and_crop_face(img, facial5point, reference_pts=reference_5pts, crop_size=output_size)
                dst_name = os.path.basename(filename)
                cv2.imwrite(os.path.join(savepath, dst_name), dst_img)

if __name__ == "__main__":
    detector = MtcnnDetector()
    filepath = './obama.jpg'
    category = 'unlabelled' # labelled or unlabelled
    process(detector, filepath, category, output_size=(112, 112), savepath='./images', visualize=True)
