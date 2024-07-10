import torch
from sklearn import metrics
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, cohen_kappa_score
# from keras.utils.np_utils import to_categorical
# from keras import utils as np_utils
# from tensorflow.python.keras.utils.np_utils import to_categorical
import torch.nn.functional as F

def metrics_score_binary(y_true, y_pred):
    """

    num_class = 1
    https://blog.csdn.net/hfutdog/article/details/88085878
    用于二分类的, metrics.py文件里面的average='micro'
    :param y_true: array[所有数据] 这里的bs是测试集或者验证集所有的图片的真是类别, 0/1
    :param y_pred: array[所有数据]  0-1 是测试集或者验证集所有的图片的预测概率
    :return:
    """
    y_label = (y_pred > 0.5).astype(float)
    acc_ = metrics.accuracy_score(y_true, y_label)
    recall_ = metrics.recall_score(y_true, y_label, average='binary')
    precision_ = metrics.precision_score(y_true, y_label, average='binary')
    # 预测结果可能全1或者全0，auc无法顺利计算
    auc_ = metrics.roc_auc_score(y_true, y_pred)  # √  使用y_pred 比y_label更合理，分数更高
    f1_ = metrics.f1_score(y_true, y_label)
    return [acc_, recall_, precision_, auc_, f1_]



def calculate_roc(y_score, y_test, n_classes):
    """
    https://blog.csdn.net/liujh845633242/article/details/102938143

    根据sklearn参考文档
    y_test是二值，y_score是概率
    :param y_score:是得到预测结果，他是概率值，并且是array
    :param y_test:是gt
    :param save_results: 保存路径
    :return:
    """
    # if n_classes == 2:
    #     y_test = to_categorical(y_test, n_classes)
    # else:
    #     # label_binarize对于两个以上的分类，可以将1维转化为多维，对于二分类，就还是一维,classes>=3才能成功使用:
    #     y_test = label_binarize(y_test, classes=list(range(n_classes)))
    y_test = F.one_hot(torch.from_numpy(y_test), num_classes=n_classes)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return fpr, tpr, roc_auc


def cm_metric(gt_labels: object, pre_prob: object, cls_num: object = 1) -> object:
    if cls_num == 1:
        pre_label = pre_prob > 0.5
        cnf_matrix = confusion_matrix(gt_labels, pre_label, labels=None, sample_weight=None)
        Accary = (cnf_matrix[1, 1] + cnf_matrix[0, 0]) / (
                cnf_matrix[1, 1] + cnf_matrix[0, 1] + cnf_matrix[0, 0] + cnf_matrix[1, 0])
        Recall = cnf_matrix[1, 1] / (cnf_matrix[1, 1] + cnf_matrix[1, 0])
        Precision = cnf_matrix[1, 1] / (cnf_matrix[1, 1] + cnf_matrix[0, 1])
        Specificity = cnf_matrix[0, 0] / (cnf_matrix[0, 1] + cnf_matrix[0, 0])

        fpr, tpr, thresholds = metrics.roc_curve(gt_labels, pre_prob)
        roc_auc = metrics.auc(fpr, tpr)
    else:
        # pre_label = torch.argmax(pre_prob, dim=1)
        pre_label = np.argmax(np.array(pre_prob), axis=1)
        cnf_matrix = confusion_matrix(gt_labels, pre_label, labels=None, sample_weight=None)

        sum_TP = 0
        for i in range(cls_num):
            sum_TP += cnf_matrix[i, i]
        Accary = sum_TP / np.sum(cnf_matrix)
        Acc_all, Recall_all, Precision_all, Specificity_all, auc_all = [], [], [], [], []
        for i in range(cls_num):
            TP = cnf_matrix[i, i]
            FP = np.sum(cnf_matrix[i, :]) - TP
            FN = np.sum(cnf_matrix[:, i]) - TP
            TN = np.sum(cnf_matrix) - TP - FP - FN
            precision = (TP / (TP + FP)) if TP + FP != 0 else 0.
            recall = (TP / (TP + FN)) if TP + FN != 0 else 0.
            specificity = (TN / (TN + FP)) if TN + FP != 0 else 0.
            Recall_all.append(recall)
            Precision_all.append(precision)
            Specificity_all.append(specificity)

        fpr, tpr, roc_auc = calculate_roc(pre_prob, gt_labels, cls_num)
        # micro：多分类；macro：计算二分类metrics的均值，为每个类给出相同权重的分值。
        Recall, Precision, Specificity, roc_auc = np.around(np.mean(Recall_all), 4), \
                                                  np.around(np.mean(Precision_all), 4), \
                                                  np.around(np.mean(Specificity_all), 4), \
                                                  np.around(roc_auc["macro"], 4)

    F1 = np.around(2 * Recall * Precision / (Recall + Precision), 4)
    kappa = cohen_kappa_score(gt_labels, pre_label)
    return Accary, Recall, Precision, Specificity, roc_auc, F1, kappa


if __name__ == '__main__':
    pred_probs = np.array(torch.randn([5,2]))
    labels = np.array([1,0,1,0,1])
    score = cm_metric(labels, pred_probs[:,1], cls_num=1)
    print(score)

    score = metrics_score_binary(labels, pred_probs[:, 1])
    print(score)


