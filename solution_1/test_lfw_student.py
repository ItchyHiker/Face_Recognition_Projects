

import numpy as np
import scipy.io
import argparse
from collections import OrderedDict

def getAccuracy(scores, flags, threshold):
    # 请根据输入来计算准确率acc的值
    '''
    scores: 配对得分
    flags: 配对是正是负
    threshold: 输入阈值
    '''
    pred_flags = np.where(scores > threshold, 1, -1)
    acc = np.sum(pred_flags == flags) / pred_flags.shape[0]
    return acc

def getThreshold(scores, flags, thrNum):
    # 请根据输入即验证集上来选取最佳阈值，目标是验证集上准确率最大时所对应的阈值平均(可存多个阈值)
    '''
    scores: 验证集配对的得分
    flags: 验证集对是正是负
    thrNum: 采样阈值间隔
    '''
    best_acc = 0
    acc_threshold_pairs = {}
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 1.0 / thrNum
    for threshold in thresholds:
        pred_flags = np.where(scores >= threshold, 1, -1)
        acc = np.sum(pred_flags == flags)
        if acc in acc_threshold_pairs.keys():
            acc_threshold_pairs[acc].append(threshold)
        else:
            acc_threshold_pairs[acc] = [threshold]

    od = OrderedDict(sorted(acc_threshold_pairs.items(), reverse=True, key=lambda t: t[0]))
    
    best_acc_threshold = od.popitem(last=False)
    
    bestThreshold = np.mean(best_acc_threshold[1])

    return bestThreshold

def evaluation_10_fold(feature_path='./lfw_result.mat'):
    ACCs = np.zeros(10)
    result = scipy.io.loadmat(feature_path)
    for i in range(10):
        fold = result['fold']
        flags = result['flag']   # 6000对样本配对情况，1表示同一个人
        featureLs = result['fl'] # 6000对左边6000样本特征
        featureRs = result['fr'] # 6000对右边6000样本特征

        valFold = fold != i
        testFold = fold == i
        flags = np.squeeze(flags)

        # 减去均值可要可不要
        mu = np.mean(np.concatenate((featureLs[valFold[0], :], featureRs[valFold[0], :]), 0), 0)
        mu = np.expand_dims(mu, 0)
        featureLs = featureLs - mu
        featureRs = featureRs - mu

        # 归一化
        featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)
        featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)

        scores = np.sum(np.multiply(featureLs, featureRs), 1)
        threshold = getThreshold(scores[valFold[0]], flags[valFold[0]], 10000)
        ACCs[i] = getAccuracy(scores[testFold[0]], flags[testFold[0]], threshold)

    return ACCs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--feature_save_path', type=str, default='./lfw_result.mat',
                        help='The path of the extract features save, must be .mat file')
    args = parser.parse_args()
    ACCs = evaluation_10_fold(args.feature_save_path)
    for i in range(len(ACCs)):
        print('{}    {:.2f}'.format(i+1, ACCs[i] * 100))
    print('--------')
    print('AVE    {:.4f}'.format(np.mean(ACCs) * 100))