'''
 Filename:  evaluate.py
 Description:  复杂指标衡量
 Created:  2022年11月3日 17时29分
 Author:  Li Shao
'''

def Evaluation4Class(prediction, y):  # 4 dim
    TP1, FP1, FN1, TN1 = 0, 0, 0, 0
    TP2, FP2, FN2, TN2 = 0, 0, 0, 0
    TP3, FP3, FN3, TN3 = 0, 0, 0, 0
    TP4, FP4, FN4, TN4 = 0, 0, 0, 0
    for i in range(len(y)):
        Act, Pre = y[i], prediction[i]
        if Act == 0 and Pre == 0: TP1 += 1
        if Act == 0 and Pre != 0: FN1 += 1
        if Act != 0 and Pre == 0: FP1 += 1
        if Act != 0 and Pre != 0: TN1 += 1
        if Act == 1 and Pre == 1: TP2 += 1
        if Act == 1 and Pre != 1: FN2 += 1
        if Act != 1 and Pre == 1: FP2 += 1
        if Act != 1 and Pre != 1: TN2 += 1
        if Act == 2 and Pre == 2: TP3 += 1
        if Act == 2 and Pre != 2: FN3 += 1
        if Act != 2 and Pre == 2: FP3 += 1
        if Act != 2 and Pre != 2: TN3 += 1
        if Act == 3 and Pre == 3: TP4 += 1
        if Act == 3 and Pre != 3: FN4 += 1
        if Act != 3 and Pre == 3: FP4 += 1
        if Act != 3 and Pre != 3: TN4 += 1
    # result
    Acc_all = round(float(TP1 + TP2 + TP3 + TP4) / float(len(y) ), 4)
    Acc1 = round(float(TP1 + TN1) / float(TP1 + TN1 + FN1 + FP1), 4)
    Prec1 = 0 if (TP1 + FP1) == 0 else round(float(TP1) / float(TP1 + FP1), 4)
    Recll1 = 0 if (TP1 + FN1) == 0 else round(float(TP1) / float(TP1 + FN1 ), 4)
    F1 = 0 if (Prec1 + Recll1) == 0 else round(2 * Prec1 * Recll1 / (Prec1 + Recll1), 4)
    Acc2 = round(float(TP2 + TN2) / float(TP2 + TN2 + FN2 + FP2), 4)
    Prec2 = 0 if (TP2 + FP2) == 0 else round(float(TP2) / float(TP2 + FP2), 4)
    Recll2 = 0 if (TP2 + FN2 ) == 0 else round(float(TP2) / float(TP2 + FN2 ), 4)
    F2 = 0 if (Prec2 + Recll2 ) == 0 else round(2 * Prec2 * Recll2 / (Prec2 + Recll2 ), 4)
    Acc3 = round(float(TP3 + TN3) / float(TP3 + TN3 + FN3 + FP3), 4)
    Prec3 = 0 if (TP3 + FP3) == 0 else round(float(TP3) / float(TP3 + FP3), 4)
    Recll3 = 0 if (TP3 + FN3) == 0 else round(float(TP3) / float(TP3 + FN3), 4)
    F3 = 0 if (Prec3 + Recll3 ) == 0 else round(2 * Prec3 * Recll3 / (Prec3 + Recll3), 4)
    Acc4 = round(float(TP4 + TN4) / float(TP4 + TN4 + FN4 + FP4), 4)
    Prec4 = 0 if (TP4 + FP4) == 0 else round(float(TP4) / float(TP4 + FP4), 4)
    Recll4 = 0 if (TP4 + FN4) == 0 else round(float(TP4) / float(TP4 + FN4), 4)
    F4 = 0 if (Prec4 + Recll4 ) == 0 else round(2 * Prec4 * Recll4 / (Prec4 + Recll4), 4)
    return Acc_all,Acc1,Prec1,Recll1,F1,Acc2,Prec2,Recll2,F2,Acc3,Prec3,Recll3,F3,Acc4,Prec4,Recll4,F4

def Evaluation2Class(prediction, y):
    TP1, FP1, FN1, TN1 = 0, 0, 0, 0
    TP2, FP2, FN2, TN2 = 0, 0, 0, 0
    for i in range(len(y)):
        Act, Pre = y[i], prediction[i]
        if Act == 0 and Pre == 0: TP1 += 1
        if Act == 0 and Pre != 0: FN1 += 1
        if Act != 0 and Pre == 0: FP1 += 1
        if Act != 0 and Pre != 0: TN1 += 1
        if Act == 1 and Pre == 1: TP2 += 1
        if Act == 1 and Pre != 1: FN2 += 1
        if Act != 1 and Pre == 1: FP2 += 1
        if Act != 1 and Pre != 1: TN2 += 1
    Acc_all = round(float(TP1+TP2)/float(len(y)), 4)
    Acc1 = round(float(TP1+TN1)/float(TP1+TN1+FN1+FP1), 4)
    Pre1 = 0 if (TP1+FP1) == 0 else round(float(TP1)/float(TP1+FP1), 4)
    Rec1 = 0 if (TP1+FN1) == 0 else round(float(TP1)/float(TP1+FN1), 4)
    F1 = 0 if (Pre1+Rec1) == 0 else round(2*Pre1*Rec1/(Pre1+Rec1), 4)
    Acc2 = round(float(TP2+TN2)/float(TP2+TN2+FN2+FP2), 4)
    Pre2 = 0 if (TP2+FP2) == 0 else round(float(TP2)/float(TP2+FP2), 4)
    Rec2 = 0 if (TP2+FN2) == 0 else round(float(TP2)/float(TP2+FN2), 4)
    F2 = 0 if (Pre2+Rec2) == 0 else round(2*Pre2*Rec2/(Pre2+Rec2), 4)
    return Acc_all,Acc1,Pre1,Rec1,F1,Acc2,Pre2,Rec2,F2

if __name__ == '__main__':
    print(1)
    