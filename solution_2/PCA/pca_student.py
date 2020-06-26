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
    # 4. 选择最大k个特征值对应特征向量组成的矩阵
    # 5. 计算投影之后的数据
    data_mat = np.array(data_mat)
    # 1. 数据中心化
    print("Begin normalizing data...")
    # print(data_mat.shape)
    M = np.mean(data_mat, axis=0)
    # print(M.shape)
    center_data = data_mat - M
    # print(center_data.shape)
    # # 2. 计算协方差矩阵
    # print("Begin calculating covriance matrix...")
    # cov_mat = np.cov(center_data.T)
    # print(cov_mat.shape)
    # # 3. 计算特征值和特征向量
    # print("Begin calculating eigen values and eigen vectors...")
    # eigen_values, eigen_vectors = np.linalg.eig(cov_mat)
    # np.save('eigen_values', eigen_values)
    # np.save('eigen_vectors', eigen_vectors)
    eigen_values = np.load('eigen_values.npy')
    eigen_vectors = np.load('eigen_vectors.npy')
    # print(eigen_values.shape)
    # print(eigen_values[:20])
    # 4. 选择最大k个特征值对应特征向量组成的矩阵
    print("Begin choosing largest eigen values and its according vectors...")
    choose_eigen_values = eigen_values[:k]
    choose_eigen_vectors = eigen_vectors[:, :k]
    # 5. 计算投影之后的数据
    print("Begin calculating data after projecting...")
    P = choose_eigen_vectors.T.dot(center_data.T)
    # print(P.shape)

    return P.T, M, choose_eigen_vectors


# test data
def test_img(img, mean_vals, re_eig_vects):
    '''
    img: 输入图像向量
    mean_vals: 预处理的均值
    re_eig_vects: 特征值向量，用来压缩数据
    return: P 经过特征向量投影后的数据
    '''
    # print("Begin PCA on test img...")
    # print(img.shape)
    # print(re_eig_vects.shape)
    mean_removed = img - mean_vals
    P = re_eig_vects.T.dot(mean_removed)
    # print(P.shape)
    return P


# compute the distance between vectors using euclidean distance
def compute_distance(vector1, vector2):
    return np.linalg.norm(np.array(vector1)[0] - np.array(vector2)[0])


# compute the distance between vectors using cosine distance
def compute_distance_(vector1, vector2):
    vector1 = vector1.reshape(1, -1)
    vector2 = vector2.reshape(1, -1)
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

    low_dim_data, mean_vals, re_eig_vects = pca_compress(data,120)
    print(low_dim_data.shape)

    # 2. use num 10 image of each person to test
    correct = 0
    for k in range(1, 41):
        img = cv.imread('orl_faces/s' + str(k) + '/10.pgm', 0)
        img = img.reshape((img.shape[0] * img.shape[1]))
        distance = test_img(img, mean_vals, re_eig_vects)
        distance_mat = []
        for i in range(1, 41):
            for j in range(1, 10):
                distance_mat.append(compute_distance_(low_dim_data[(i - 1) * 9 + j - 1, :], distance.reshape((1, -1))))
        num_ = np.argmax(distance_mat)
        class_ = int(np.argmax(distance_mat) / 9) + 1
        if class_ == k:
            correct += 1
        print('s' + str(k) + '/10.pgm is the most similar to s' +
              str(class_) + '/' + str(num_ % 9 + 1) + '.pgm')
    print("accuracy: %lf" % (correct / 40))
