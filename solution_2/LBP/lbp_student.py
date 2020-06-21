# import the necessary packages
import numpy as np
import cv2

def original_lbp(image):
    """origianl local binary pattern"""
    rows = image.shape[0]
    cols = image.shape[1]
    lbp_image = np.zeros((rows - 2, cols - 2), np.uint8)

    # 计算每个像素点的lbp值，具体范围如上lbp_image

    return lbp_image

if __name__ == '__main__':
    image = cv2.imread("./lms.jpg", 0)
    cv2.imshow("image", image)
    org_lbp_image = original_lbp(image)
    cv2.imshow("org_lbp_image", org_lbp_image)
    cv2.waitKey()