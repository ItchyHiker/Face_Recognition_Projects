import cv2 as cv
import numpy as np

# https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/

# train data
def pca_compress(data_mat, k=9999999):
    '''
    :param data_mat: 输入数据
    :param k:
    :return:
    '''
    # 1. 数据中心化
    # 2. 计算协方差矩阵
    # 3. 计算特征值和特征向量
    # 4.
    # 5.
    data_mat = np.array(data_mat)
    print(data_mat.shape)
    M = np.mean(data_mat, axis=0)
    print(M.shape)
    center_data = data_mat - M
    print(center_data.shape)
    cov_mat = np.cov(center_data.T)
    print(cov_mat.shape)
    eigen_values, engen_vectors = np.linalg.eig(cov_mat)

    P = data_mat.T.dot(T)
    print(P)
    return low_dim_data, mean_vals, re_eig_vects


# test data
def test_img(img, mean_vals, low_dim_data):
    mean_removed = img - mean_vals
    return mean_removed * low_dim_data.T


# compute the distance between vectors using euclidean distance
def compute_distance(vector1, vector2):
    return np.linalg.norm(np.array(vector1)[0] - np.array(vector2)[0])


# compute the distance between vectors using cosine distance
def compute_distance_(vector1, vector2):
    return np.dot(np.array(vector1)[0], np.array(vector2)[0]) / (np.linalg.norm(np.array(vector1)[0]) * (np.linalg.norm(np.array(vector2)[0])))


if __name__ == '__main__':

    # 1. use num 1- 9 image of each person to train
    data = []
    for i in range(1, 41):
        for j in range(1, 10):
            img = cv.imread('orl_faces/s' + str(i) + '/' + str(j) + '.pgm', 0)
            width, height = img.shape
            img = img.reshape((img.shape[0] * img.shape[1]))
            data.append(img)

    low_dim_data, mean_vals, re_eig_vects = pca_compress(data, 120)


    # 2. use num 10 image of each person to test
    correct = 0
    for k in range(1, 41):
        img = cv.imread('orl_faces/s' + str(k) + '/10.pgm', 0)
        img = img.reshape((img.shape[0] * img.shape[1]))
        distance = test_img(img, mean_vals, low_dim_data)
        distance_mat = []
        for i in range(1, 41):
            for j in range(1, 10):
                distance_mat.append(compute_distance_(re_eig_vects[(i - 1) * 9 + j - 1], distance.reshape((1, -1))))
        num_ = np.argmax(distance_mat)
        class_ = int(np.argmax(distance_mat) / 9) + 1
        if class_ == k:
            correct += 1
        print('s' + str(k) + '/10.pgm is the most similar to s' +
              str(class_) + '/' + str(num_ % 9 + 1) + '.pgm')
    print("accuracy: %lf" % (correct / 40))
