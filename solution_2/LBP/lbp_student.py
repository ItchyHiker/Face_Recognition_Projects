# import the necessary packages
import numpy as np
import cv2

def original_lbp(image):
    """origianl local binary pattern"""
    rows = image.shape[0]
    cols = image.shape[1]
    lbp_image = np.zeros((rows - 2, cols - 2), np.uint8)

    # 计算每个像素点的lbp值，具体范围如上lbp_image
    binary_values = [0]*8
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            binary_values[0] = 0 if (image[i, j] > image[i-1, j-1]) else 1
            binary_values[1] = 0 if (image[i, j] > image[i-1, j]) else 1
            binary_values[2] = 0 if (image[i, j] > image[i-1, j+1]) else 1
            binary_values[3] = 0 if (image[i, j] > image[i, j-1]) else 1
            binary_values[4] = 0 if (image[i, j] > image[i, j+1]) else 1
            binary_values[5] = 0 if (image[i, j] > image[i+1, j-1]) else 1
            binary_values[6] = 0 if (image[i, j] > image[i+1, j]) else 1
            binary_values[7] = 0 if (image[i, j] > image[i+1, j+1]) else 1
            # print(int("".join(str(x) for x in binary_values),2))
            lbp_image[i-1, j-1] = int("".join(str(x) for x in binary_values),2)

    return lbp_image

if __name__ == '__main__':
    image = cv2.imread("./lms.jpg", 0)
    cv2.imshow("image", image)
    org_lbp_image = original_lbp(image)
    cv2.imshow("org_lbp_image", org_lbp_image)
    cv2.waitKey()