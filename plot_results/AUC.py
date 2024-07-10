import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d

if __name__ == '__main__':
    # path = r'D:\code\assistance\12_jiangbo_MIL\completed_experiments\plot\AMD'
    # # TinyViT = ['attention-based-resnet18', 'dsmil', 'loss-attention-res18', 'MS-DA-MIL', 'CLAM', 'MIL-VT' ,'ours', 'ours-val_acc-1st', 'ours-val_acc-2nd', 'ours-val_acc-3rd', 'AMD-CLAM-val_acc']
    # TinyViT = ['attention-based-resnet18', 'dsmil', 'loss-attention-res18', 'MS-DA-MIL', 'MIL-VT', 'CLAM' ,'ours-val_acc-3rd']
    from sklearn import metrics
    import seaborn as sns
    import matplotlib.colors as mcolors

    dataset_factory = 'oriR2_split_fileR1'  #
    root = '/disk1/imed_hjq/code/3-PD_analysis/02_baseline/checkpoints/csv_pro/'

    models = os.listdir(root)
    # TinyViT = [
    #     'res18_oh_20230410R1_oriR2_split_fileR1_res18.csv',
    #     'res50_oh_20230411R1_oriR2_split_fileR1_res50.csv',
    #     'se_res18_oh_20230419R1_oriR2_split_fileR1_se_res18.csv',
    #     'cbam_resnet50_oh_20230601R2_oriR2_split_fileR1_cbam_resnet50.csv',
    #     'coord_resnet18_oh_20230411R1_oriR2_split_fileR1_coord_resnet18.csv',
    #     'fcanet50_oh_20230601R2_oriR2_split_fileR1_fcanet50.csv',
    #     'ours7_res50_oh_20230605R5_oriR2_split_fileR1_ours7_res50.csv',
    # ]
    # name_dict = {
    #     "res18": "ResNet18",
    #     "res50": "ResNet50",
    #     "se": "ResNet18+SE",
    #     "cbam": "ResNet50+CBAM",
    #     "coord": "ResNet18+Coord",
    #     "fcanet50": "ResNet50+Fca",
    #     "ours7": "ResNet50+SRNet",
    # }
    models = [
        'wavepooling_oh_20230601R2_oriR2_split_fileR1_wavepooling.csv',
        'wcnn_oh_20230601R2_oriR2_split_fileR1_wcnn.csv',
        'dawn_oh_20230601R2_oriR2_split_fileR1_dawn.csv',
        'wavecnet_oh_20230601R2_oriR2_split_fileR1_wavecnet.csv',
        'ours7back_oh_20230602R1_oriR2_split_fileR1_ours7back.csv',
        'ours7_res50_oh_20230605R5_oriR2_split_fileR1_ours7_res50.csv',
    ]
    name_dict = {
        "ours7back": "ResNet18+SRM",
        "ours7": "ResNet50+SRM",
        "wcnn": "WCNN",
        "wavepooling": "WavePooling",
        "dawn": "DAWN",
        "wavecnet": "WaveCNet",
    }

    # palette = sns.color_palette("deep", len(TinyViT))  # gist_rainbow gist_ncar hsv Set1
    # colors = [mcolors.rgb2hex(color) for color in palette]
    # print(colors)
    # asd
    # colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3', '#937860', '#da8bc3']
    colors = ['#dd8452', '#55a868', '#c44e52', '#8172b3', '#937860', '#da8bc3']

    prob = [], []

    plt.cla()
    plt.close('all')
    # fig, ax = plt.subplots(figsize=(4, 4))
    fig, ax = plt.subplots()
    kk = 0

    for i, file in enumerate(models):
        data = pd.read_csv(root + '/' + file)

        fpr, tpr, thresholds = metrics.roc_curve(data['gt'].values, data['pro'].values)
        roc_auc = metrics.auc(fpr, tpr)

        aaa = float('%.4f' % roc_auc) * 100
        ax.plot(fpr, tpr,
                label=name_dict[file.split('_')[0]] + ' (area = {0:0.4f})'
                                                      ''.format(roc_auc),
                color=colors[kk], linestyle='-', linewidth=2)
        kk += 1

        # plt.title('REFUGE dataset', fontsize=20)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.tick_params(labelsize=18)
        plt.xlabel('FPR', fontsize=18)  # False Positive Rate
        plt.ylabel('TPR', fontsize=18)  # True Positive Rate
        # plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right", prop={'size': 10})
        plt.rc('font', family='Times New Roman')

    # plt.savefig(os.path.join(root, '../plot/att_based_roc.pdf'), format="pdf",
    #             bbox_inches='tight', pad_inches=0.1)
    plt.savefig(os.path.join(root, '../plot/wavelet_based_roc.pdf'), format="pdf",
                bbox_inches='tight', pad_inches=0.1)
    plt.show()
