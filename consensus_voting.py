import pandas as pd
import os, sys
import numpy as np

# train test
table1 = pd.read_csv('pah_prediction_a4c.csv', index_col=0)
table2 = pd.read_csv('pah_prediction_plax.csv', index_col=0)
info = pd.read_csv('train_test_pah_a4c_plax_all.csv', index_col=0)
train_index = pd.read_csv('train_index.csv', header=None)[0].tolist()
test_index = pd.read_csv('test_index.csv', header=None)[0].tolist()
# info=info.loc[train_index,:]
info = info.loc[test_index, :]


# #val
# table1=pd.read_csv('pah_prediction_a4c_val.csv',index_col=0)
# table2=pd.read_csv('pah_prediction_plax_val.csv',index_col=0)
# info=pd.read_csv('../echotrans-gpu/train_test_pah_normal_a4c_plax.csv',index_col=0)

def score(patientid):
    a4c = eval(info.loc[patientid, 'a4c'])
    plax = eval(info.loc[patientid, 'plax'])
    if a4c and plax:
        s = 0
        for im in a4c:
            s += table1.loc[im, 'PAH_pre']
        aver_s = s / len(a4c)
        t = 0
        for img in plax:
            t += table2.loc[img, 'PAH_pre']
        aver_t = t / len(plax)
        # print(aver_s,' for a4c and ',aver_t,' for plax')
        return aver_s, aver_t
    elif a4c:
        s = 0
        for im in a4c:
            s += table1.loc[im, 'PAH_pre']
        aver_s = s / len(a4c)
        aver_t = aver_s
        # print(aver_s,' for a4c only')
        return aver_s, aver_t
    elif plax:
        t = 0
        for img in plax:
            t += table2.loc[img, 'PAH_pre']
        aver_t = t / len(plax)
        # print(aver_t, ' for plax only')
        aver_s = aver_t
        return aver_s, aver_t
    else:
        # print('Neither a4c or plax obtained')
        return None, None


def diagnose(patientid):
    a4c_score, plax_score = score(patientid)
    if a4c_score == None:
        # print('None')
        return None, None
    elif a4c_score >= 0.5 and plax_score >= 0.5:
        # print('diagnosed: Positive')
        return 1, 1
    elif a4c_score < 0.5 and plax_score < 0.5:
        # print('diagnosed: Negative')
        return 0, 0
    elif a4c_score >= 0.5:
        # print('uncertain')
        return 1, 0
    elif plax_score >= 0.5:
        # print('uncertain')
        return 0, 1


def TFPN(gts, preds):
    gts = np.array(gts)
    preds = np.array(preds)
    TP = np.dot(gts, preds)
    FN = np.dot(gts, preds * (-1) + 1)
    FP = np.dot(gts * (-1) + 1, preds)
    TN = np.dot(gts * (-1) + 1, preds * (-1) + 1)
    recall = TP / (TP + FN)
    TPR = TP / (TP + FP)
    TNR = TN / (TN + FN)
    accuracy = (TP + TN) / len(preds)
    precision = TP / (TP + FP)
    specificity = TN / (TN + FP)
    F1_measure = 2 * precision * recall / (precision + recall)
    return round(accuracy, 3), round(TPR, 3), round(TNR, 3)


def two_experts_TFPN(gts, preds1, preds2):
    gts = np.array(gts)
    preds1 = np.array(preds1)
    preds2 = np.array(preds2)
    TPTP = np.dot(np.multiply(gts, preds1), preds2)
    TPFN = np.dot(np.multiply(gts, preds1), preds2 * (-1) + 1)
    FNTP = np.dot(np.multiply(gts, preds2), preds1 * (-1) + 1)
    FNFN = np.dot(np.multiply(gts, preds2 * (-1) + 1), preds1 * (-1) + 1)
    TNFP = np.dot(np.multiply(gts * (-1) + 1, preds2), preds1 * (-1) + 1)
    FPTN = np.dot(np.multiply(gts * (-1) + 1, preds1), preds2 * (-1) + 1)
    FPFP = np.dot(np.multiply(gts * (-1) + 1, preds1), preds2)
    TNTN = np.dot(np.multiply(gts * (-1) + 1, preds1 * (-1) + 1), preds2 * (-1) + 1)
    acc = (TPTP + TNTN) / sum([TPTP, TPFN, FNTP, FNFN, TNTN, TNFP, FPTN, FPFP])
    tpr = TPTP / (TPTP + FPFP)
    tnr = TNTN / (TNTN + FNFN)
    return round(acc, 3), round(tpr, 3), round(tnr, 3)


if __name__ == '__main__':
    grts = info['PAH']
    grts[grts > 0] = 1
    gts = []
    preds1 = []
    preds2 = []

    index = table1.index.intersection(table2.index).intersection(grts.index)
    preds1 = table1.loc[index, 'PAH_pre']
    preds2 = table2.loc[index, 'PAH_pre']
    gts = grts[index]
    acc22, tpr22, tnr22 = two_experts_TFPN(gts, preds1, preds2)

    acc1, tpr1, tnr1 = TFPN(gts, preds1)
    acc2, tpr2, tnr2 = TFPN(gts, preds2)
    res = pd.DataFrame([[acc1, tpr1, tnr1], [acc2, tpr2, tnr2], [acc22, tpr22, tnr22]], columns=['acc', 'tpr', 'tnr'],
                       index=['a4c only', 'plax only', 'joint'])
    # res.to_csv('val_TFPN.csv')
    res.to_csv('test_TFPN.csv')
