import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

import pickle
import utils.imgPreprocess as imgPreprocess
import utils.CustomDataset as CustomDataset
from utils.sortModelList import sortModelList
from utils import featureExtract
from utils.distanceAverage import distanceAverage

from collections import defaultdict

import models.convnext as convnext
import models.convnextAttention as convnextAttention
import GCENet.GCENetWithECA as gcenet


def test(args, Fold):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.__version__)
    print(device)

    testDBPath = args.DBPath + '\\' + args.inputDB + '\\' + Fold + '\\test\\allcam'
    numOfClass = len(os.listdir(args.DBPath + '\\' + args.inputDB + '\\' + Fold + '\\train\\allcam'))
    print("numOfClass :", numOfClass)
    print('Fold : ' + Fold)

    GCENet = gcenet.GCENet(inputChannel=384 * 2 + 1 + 1024)
    GCENet.fc = nn.Linear(384 * 2 + 1 + 1024, numOfClass)
    oriModel = convnext.convnext_micro(num_classes=numOfClass)
    attModel = convnextAttention.convnext_micro(num_classes=numOfClass)

    oriPATH = './' + args.inputDB + '/valResult/' + 'original/' + \
           Fold + '/selectEpoch'
    oriModelPaths = os.listdir(oriPATH)
    oriModelPaths = sortModelList(oriModelPaths)

    attPATH = './' + args.inputDB + '/valResult/' + 'attention/' + \
           Fold + '/selectEpoch'
    attModelPaths = os.listdir(attPATH)
    attModelPaths = sortModelList(attModelPaths)

    gcePATH = './' + args.inputDB + '/valResult/' + 'gceNet/' + \
           Fold + '/selectEpoch'
    gceModelPaths = os.listdir(gcePATH)
    gceModelPaths = sortModelList(gceModelPaths)

    trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    testSet = CustomDataset.CustomDataset(root_dir=testDBPath, transforms=trans)
    loader = DataLoader(testSet, batch_size=args.testBatchSize, shuffle=False, pin_memory=True)

    for i, gceModelPath in enumerate(gceModelPaths):
        ReIDdict = {}
        progress = 0

        oriModel.load_state_dict(torch.load(oriPATH + '\\' + oriModelPaths[i]), strict=False)
        oriModel.eval()
        oriModel.to(device)

        attModel.load_state_dict(torch.load(attPATH + '\\' + attModelPaths[i]), strict=False)
        attModel.eval()
        attModel.to(device)

        GCENet.load_state_dict(torch.load(gcePATH + '\\' + gceModelPath), strict=False)
        GCENet.eval()
        GCENet.to(device)

        saveFeature = featureExtract.saveFeature()
        oriModel.stages[-1].register_forward_hook(saveFeature.saveOriFeature)
        attModel.stages[-1].register_forward_hook(saveFeature.saveAttFeature)
        attModel.stages[-1][1].SpatialGate.spaSig.register_forward_hook(saveFeature.saveCBAMSpatialFeature)
        imgSplit = imgPreprocess.imgPreprocess().imgSplit

        epochNum = gceModelPaths[i].split('_')[-1].split('.')[0]
        with torch.no_grad():
            ReIDresult = []

            LoaderIter = iter(loader)
            pbar = tqdm(range(len(loader)))
            pbar.set_description(f'Epoch - {epochNum}')
            for pbar_i in pbar:
                data = next(LoaderIter)
                galleryLabels = []
                gallerySaveNames = []
                galleryImgs = data['image'].to(device)
                galleryNames = data['filename']

                inputImgs = galleryImgs

                ## GCE
                oriModel.wsFlag = False
                attModel.wsFlag = False
                saveFeature.initFeature()
                _, _, _ = oriModel(inputImgs)
                _, _, _ = attModel(inputImgs)
                patchImg = imgSplit(img=inputImgs, batch=inputImgs.size(dim=0))
                oriFeature = distanceAverage(saveFeature.returnFeature()[0][0], 384, inputImgs.size(dim=0))
                attFeature = distanceAverage(saveFeature.returnFeature()[1][0], 384, inputImgs.size(dim=0))
                inputFeature = torch.cat([oriFeature, attFeature, saveFeature.returnFeature()[2][0], patchImg], dim=1)

                outputs, gceFeature, _ = GCENet(inputFeature)

                for names in galleryNames:
                    nameParsing = names.split('\\')
                    galleryLabels.append(nameParsing[len(nameParsing)-2])
                    gallerySaveNames.append(nameParsing[len(nameParsing)-1])

                for result_i in range(len(gceFeature)):
                    ReIDresult.append([galleryLabels[result_i], gallerySaveNames[result_i], gceFeature[result_i]])

                ReIDdict = ReIDresult

        reidResult = defaultdict(list)

        if args.inputDB == 'SYSU-MM01_thermal':
            #center image 정하기
            centerImg = []
            imgTmp = []
            calTmp = []

            keyTmp = 0
            for i, (key_cen, name_cen, value_cen) in enumerate(ReIDdict):
                key_cen = int(key_cen)
                if keyTmp != 0 and keyTmp != key_cen:
                    for cen_i, (key_tmp, name_tmp, value_tmp) in enumerate(imgTmp):
                        l2AllVal = 0
                        for cen2_i, (key_tmp2, name_tmp2, value_tmp2) in enumerate(imgTmp):
                            if name_tmp == name_tmp2:
                                continue

                            l2Value = torch.dist(value_tmp, value_tmp2, 2)

                            l2AllVal += l2Value
                        calTmp.append([key_tmp, name_tmp, value_tmp, l2AllVal])
                    calTmp.sort(key=lambda x: x[3])
                    centerImg.append([calTmp[0][0], calTmp[0][1], calTmp[0][2]])
                    calTmp = []
                    imgTmp = []
                keyTmp = key_cen
                imgTmp.append([key_cen, name_cen, value_cen])

            if imgTmp != []:
                for cen_i, (key_tmp, name_tmp, value_tmp) in enumerate(imgTmp):
                    l2AllVal = 0
                    for cen2_i, (key_tmp2, name_tmp2, value_tmp2) in enumerate(imgTmp):
                        if name_tmp == name_tmp2:
                            continue

                        l2Value = torch.dist(value_tmp, value_tmp2, 2)

                        l2AllVal += l2Value
                    calTmp.append([key_tmp, name_tmp, value_tmp, l2AllVal])
                calTmp.sort(key=lambda x: x[3])
                centerImg.append([calTmp[0][0], calTmp[0][1], calTmp[0][2]])

            for i, (key_gal, name_gal, value_gal) in enumerate(centerImg):
                key_gal = int(key_gal)
                print('gal_cnt : ', i)
                for j, (key_prob, name_prob, value_prob) in enumerate(ReIDdict):
                    key_prob = int(key_prob)
                    if key_gal == key_prob and name_gal == name_prob:
                        continue

                    value_cal = torch.dist(value_gal, value_prob, 2)

                    reidResult[key_gal].append([key_prob, name_prob, value_cal])

        else:
            for i, (key_gal, name_gal, value_gal) in enumerate(ReIDdict):
                key_gal = int(key_gal)
                for j, (key_prob, name_prob, value_prob) in enumerate(ReIDdict):
                    key_prob = int(key_prob)
                    if key_gal == key_prob and name_gal == name_prob:
                        continue

                    value_cal = torch.dist(value_gal, value_prob, 2)

                    reidResult[key_gal].append([key_prob, name_prob, value_cal])

        for key_gal in reidResult.keys():
            reidResult[key_gal].sort(key=lambda x: x[2])

        ReIDdict = reidResult

        saveTestResultName = gceModelPath.split('.')[0]
        testPath = './' + args.inputDB + '\\' + 'testResult/' + 'gceNet' + '\\' + Fold + '/epochTermTest/'
        if not os.path.isdir(testPath):
            os.makedirs(testPath)
        with open(testPath + 'ReID_test_result_' + saveTestResultName + '.pickle',
                  'wb') as fw:
            pickle.dump(ReIDdict, fw)

    del testSet
    del loader
    torch.cuda.empty_cache()