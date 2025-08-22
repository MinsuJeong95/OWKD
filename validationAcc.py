import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import shutil
from collections import defaultdict
from utils.findTopFeature import findTopFeature
from utils.sortModelList import sortModelList


def mySoftmax(a) :
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def accCalculate(args, Fold):
    oriPATH = '.\\' + args.inputDB + '\\valResult\\' + 'original' + '\\' + Fold
    attPATH = '.\\' + args.inputDB + '\\valResult\\' + 'attention' + '\\' + Fold
    gcePATH = '.\\' + args.inputDB + '\\valResult\\' + 'gceNet' + '\\' + Fold

    pickleFilePath = os.listdir(gcePATH + '\\epochTermValidation')

    # 피클list
    modelNum = []
    for i in range(len(pickleFilePath)):  # 피클list 재배열
        modelName = pickleFilePath[i].split('_')
        modelNum.append(int(modelName[len(modelName) - 1].split('.')[0]))
    modelNum.sort()
    for num_i in range(len(modelNum)):  # 재배열 적용
        modelName = pickleFilePath[num_i].split('_')
        del modelName[len(modelName) - 1]
        modelName.append(str(modelNum[num_i]) + '.pickle')
        saveName = ''
        for name_i in range(len(modelName)):  # 나눠져 있는 list 재 연결
            if name_i == (len(modelName) - 1):
                saveName = saveName + modelName[name_i]
            else:
                saveName = saveName + modelName[name_i] + '_'
        pickleFilePath[num_i] = saveName

    print(len(pickleFilePath))

    Accuracy = []
    Accuracy.append(0)
    lossArray = []
    lossArray.append(60)
    for fileCnt in range(len(pickleFilePath)):
        print(fileCnt)
        with open(gcePATH + '\\epochTermValidation'+'/'+pickleFilePath[fileCnt], 'rb') as fr:
            loadReIDdict = pickle.load(fr)
        ReIDdict = loadReIDdict

        folderPath = gcePATH + '\\valResultGraph' + '/' + pickleFilePath[fileCnt].split('.')[0]
        try:
            if not os.path.exists(folderPath):
                os.makedirs(folderPath)
        except OSError:
            print('Error')

        f = open(folderPath+"/uncorrected" + ".txt", 'w')

        TP = 0
        TN = 0
        FP = 0
        FN = 0
        Recall = 0
        Precision = 0
        totalLoss = 0
        totalScore = 0
        valueCnt = 0
        allReIDValue = []
        allFeatureValues = []
        allLabels = []
        distanceFeature = defaultdict(list)
        accTmp = 0

        for i, (key, value) in enumerate(ReIDdict.items()):
            if key == 0:
                continue
            for valueData in value:
                allFeatureValues.append(valueData[0])
                allLabels.append(key)

                totalLoss += valueData[1]
                valueCnt += 1

        for i, (key, value) in enumerate(ReIDdict.items()):
            features = []
            if key == 0:
                continue
            print(key)
            for valueData in value:
                features.append(valueData[0])
            topFeature = findTopFeature(features)

            for feature_i, ReIDValue in enumerate(allFeatureValues):
                result = torch.dist(topFeature, ReIDValue, 2)
                if result == 0:
                    continue
                distanceFeature[key].append((result, allLabels[feature_i]))

        for i, (key, values) in enumerate(distanceFeature.items()):
            TP = 0
            FP = 0
            values.sort(key=lambda x: x[0])
            for value in values:
                if key == value[1]:
                    TP = TP + 1
                elif key != value[1]:
                    FP = FP + 1

                if TP >= (len(ReIDdict[key])-1):
                    break
            accTmp = accTmp + ((len(values)-FP) / len(values))


        acc = accTmp / len(distanceFeature)
        Accuracy.append(acc*100)
        print('Accuracy : ', acc)
        f.write('Accuracy : ' + str(acc) + '\n')

        loss = totalLoss / valueCnt
        lossArray.append(loss)
        print('Loss : ', loss)
        f.write('Loss : ' + str(loss) + '\n')

        plt.close()
        f.close()

    ####################
    plt.figure(0)
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.axis([0, len(pickleFilePath), 0, 100.5])
    ax1.grid(True)
    valAccGraph = ax1.plot(range(0, len(pickleFilePath)+1), Accuracy, color='C0', label='Accuracy (validation) ', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=16)
    ax1.set_ylabel('Accuracy (%)', fontsize=16)

    ax2.axis([0, len(pickleFilePath), 0, 60])
    valLossGraph = ax2.plot(range(0, len(pickleFilePath)+1), lossArray, color='C1', label='Loss (validation)', linewidth=2)
    ax2.set_ylabel('Loss', fontsize=16)

    lgd = valAccGraph + valLossGraph
    labs = [l.get_label() for l in lgd]
    ax1.legend(lgd, labs, loc='center left', bbox_to_anchor=(0.5, 0.5), ncol=1, prop={'size': 12})
    # ax2.xaxis.set_major_locator(FixedLocator(np.arange(0, trainDataLen + 1, trainDataLen / 20)))
    # ax2.xaxis.set_major_formatter(FixedFormatter(np.arange(0, numEpoch + 1, int(numEpoch / 20))))
    ###################

    foldType = Fold

    folderPath = '.\\' + args.inputDB + '\\valResult\\' + 'gceNet' + '\\' + Fold + '\\valResultGraph'
    if not os.path.isdir(folderPath):
        os.makedirs(folderPath)
    plt.savefig(folderPath + '/Val_' + foldType + '_' + 'gceNet' + '.png')
    plt.close()

    np.save(folderPath + '/Val_acc_' + foldType + '_' + 'gceNet', np.array(Accuracy))
    np.save(folderPath + '/Val_loss_' + foldType + '_' + 'gceNet', np.array(lossArray))

    # 모델 select
    Accuracy.pop(0)
    accMax = max(Accuracy)
    modelSelect = 0
    for i in range(len(Accuracy)):
        if Accuracy[i] == accMax:
            modelSelect = i
            break

    trainPath = '.\\' + args.inputDB + '\\trainModels\\' + 'original' + '\\' + Fold + '\epochTermModel'
    modelPath = os.listdir(trainPath)
    oriModelPath = sortModelList(modelPath)
    oriSelectTrainPath = oriPATH + '\\selectEpoch\\'
    if not os.path.isdir(oriSelectTrainPath):
        os.makedirs(oriSelectTrainPath)
    shutil.copyfile(trainPath + '\\' + oriModelPath[modelSelect], oriSelectTrainPath + oriModelPath[modelSelect])

    trainPath = '.\\' + args.inputDB + '\\trainModels\\' + 'attention' + '\\' + Fold + '\epochTermModel'
    modelPath = os.listdir(trainPath)
    attModelPath = sortModelList(modelPath)
    attSelectTrainPath = attPATH + '\\selectEpoch\\'
    if not os.path.isdir(attSelectTrainPath):
        os.makedirs(attSelectTrainPath)
    shutil.copyfile(trainPath + '\\' + attModelPath[modelSelect], attSelectTrainPath + attModelPath[modelSelect])

    trainPath = '.\\' + args.inputDB + '\\trainModels\\' + 'gceNet' + '\\' + Fold + '\epochTermModel'
    modelPath = os.listdir(trainPath)
    gceModelPath = sortModelList(modelPath)
    gceSelectTrainPath = gcePATH + '\\selectEpoch\\'
    if not os.path.isdir(gceSelectTrainPath):
        os.makedirs(gceSelectTrainPath)
    shutil.copyfile(trainPath + '\\' + gceModelPath[modelSelect], gceSelectTrainPath + gceModelPath[modelSelect])
