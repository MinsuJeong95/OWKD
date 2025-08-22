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
from utils.choiceData import choiceData
import kdLoss.kd as kd

import models.convnext as convnext
import models.convnextAttention as convnextAttention
import GCENet.GCENetWithECA as gcenet

import kdLoss.OWKD as OWKD


def validation(args, Fold):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.__version__)
    print(device)

    validationDBPath = args.DBPath + '\\' + args.inputDB + '\\' + Fold + '\\validation\\allcam'
    numOfClass = len(os.listdir(args.DBPath + '\\' + args.inputDB + '\\' + Fold + '\\train\\allcam'))
    print("numOfClass :", numOfClass)
    print('Fold : ' + Fold)

    GCENet = gcenet.GCENet(inputChannel=384 * 2 + 1 + 1024)
    GCENet.fc = nn.Linear(384 * 2 + 1 + 1024, numOfClass)
    oriModel = convnext.convnext_micro(num_classes=numOfClass)
    attModel = convnextAttention.convnext_micro(num_classes=numOfClass)
    optimalParam = kd.kdlossOptimizing()

    oriPATH = './' + args.inputDB + '/trainModels/' + 'original/' + \
           Fold + '/epochTermModel'
    oriModelPaths = os.listdir(oriPATH)
    oriModelPaths = sortModelList(oriModelPaths)

    attPATH = './' + args.inputDB + '/trainModels/' + 'attention/' + \
           Fold + '/epochTermModel'
    attModelPaths = os.listdir(attPATH)
    attModelPaths = sortModelList(attModelPaths)

    gcePATH = './' + args.inputDB + '/trainModels/' + 'gceNet/' + \
           Fold + '/epochTermModel'
    gceModelPaths = os.listdir(gcePATH)
    gceModelPaths = sortModelList(gceModelPaths)

    lossPATH = './' + args.inputDB + '/trainModels/' + 'lossModel/' + \
           Fold + '/epochTermModel'
    lossModelPaths = os.listdir(lossPATH)
    lossModelPaths = sortModelList(lossModelPaths)

    teacherOri = convnext.convnext_small(pretrained=True, num_classes=numOfClass)
    teacherAtt = convnextAttention.convnext_small(pretrained=True, num_classes=numOfClass)
    teacherGCE = gcenet.GCENet(inputChannel=384 * 2 + 1 + 1024)
    teacherGCE.fc = nn.Linear(384 * 2 + 1 + 1024, numOfClass)

    teacherOriPath = 'kdLoss/WSEmodel/' + args.inputDB + '/bestModel/original/' + Fold
    teacherOriWeight = os.listdir(teacherOriPath)
    teacherOri.load_state_dict(torch.load('./' + teacherOriPath + '/' + teacherOriWeight[0]), strict=False)
    teacherOri.eval()
    teacherOri.to(device)

    teacherAttPath = 'kdLoss/WSEmodel/' + args.inputDB + '/bestModel/attention/' + Fold
    teacherAttWeight = os.listdir(teacherAttPath)
    teacherAtt.load_state_dict(torch.load('./' + teacherAttPath + '/' + teacherAttWeight[0]), strict=False)
    teacherAtt.eval()
    teacherAtt.to(device)

    teacherGCEPath = 'kdLoss/WSEmodel/' + args.inputDB + '/bestModel/gce/' + Fold
    teacherGceWeight = os.listdir(teacherGCEPath)
    teacherGCE.load_state_dict(torch.load('./' + teacherGCEPath + '/' + teacherGceWeight[0]), strict=False)
    teacherGCE.eval()
    teacherGCE.to(device)

    trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    valSet = CustomDataset.CustomDataset(root_dir=validationDBPath, transforms=trans)
    loader = DataLoader(valSet, batch_size=args.valBatchSize, shuffle=True, pin_memory=True)

    margin = 20.0
    cnt = 0
    distillFlag = True

    for i, gceModelPath in enumerate(gceModelPaths):
        ReIDdict = {}
        progress = 0

        oriModel.load_state_dict(torch.load(oriPATH + '\\' + oriModelPaths[i]))
        oriModel.eval()
        oriModel.to(device)

        attModel.load_state_dict(torch.load(attPATH + '\\' + attModelPaths[i]))
        attModel.eval()
        attModel.to(device)

        GCENet.load_state_dict(torch.load(gcePATH + '\\' + gceModelPath))
        GCENet.eval()
        GCENet.to(device)

        optimalParam.load_state_dict(torch.load(lossPATH + '\\' + lossModelPaths[i]))
        optimalParam.eval()
        optimalParam.to(device)

        saveFeature = featureExtract.saveFeature()
        oriModel.stages[-1].register_forward_hook(saveFeature.saveOriFeature)
        attModel.stages[-1].register_forward_hook(saveFeature.saveAttFeature)
        attModel.stages[-1][1].SpatialGate.spaSig.register_forward_hook(saveFeature.saveCBAMSpatialFeature)
        imgSplit = imgPreprocess.imgPreprocess().imgSplit

        cnt += 1
        if cnt % 15 == 0:
            margin -= 6
        criterionTriplet = nn.TripletMarginLoss(margin=margin).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        OWKDLoss = OWKD.OWKD_loss(device)

        epochNum = gceModelPaths[i].split('_')[-1].split('.')[0]
        with torch.no_grad():
            ReIDresult = []
            keyTmp = 0

            LoaderIter = iter(loader)
            pbar = tqdm(range(len(loader)))
            pbar.set_description(f'Epoch - {epochNum}')
            for pbar_i in pbar:
                data = next(LoaderIter)
                labels = data['label'].to(device)
                gallerySaveNames = []
                gallerySaveLabels = []
                galleryImgs = data['image'].to(device)

                galleryNames = data['filename']

                for i in range(len(galleryNames)):
                    galleryRealName = galleryNames[i].split('\\')
                    galleryRealNameLen = len(galleryRealName)
                    galleryRealLabel = galleryRealName[galleryRealNameLen - 2]
                    gallerySaveName = galleryRealName[galleryRealNameLen - 3] + '_' + \
                                      galleryRealName[galleryRealNameLen - 2] + '_' + \
                                      galleryRealName[galleryRealNameLen - 1].split('.')[0]

                    gallerySaveNames.append(gallerySaveName)
                    gallerySaveLabels.append(galleryRealLabel)

                inputImgs = galleryImgs

                teacherOriFeature, teacherAttFeature, teacherGCEFeature, teacherOriOutPuts, teacherAttOutPuts, teacherGCEOutPuts = \
                    kd.extractTeacherFeature(teacherOri, teacherAtt, teacherGCE, inputImgs)

                ## GCE
                oriModel.wsFlag = False
                oriModel.multiFeature = False
                attModel.wsFlag = False
                attModel.multiFeature = False
                saveFeature.initFeature()
                oriOutputs, oriFeature, oriLayerFeature = oriModel(inputImgs)
                anchor, positive, negative = choiceData(labels, oriFeature, device)
                oriTpLoss = criterionTriplet(anchor, positive, negative)

                attOutputs, attFeature, attLayerFeature = attModel(inputImgs)
                anchor, positive, negative = choiceData(labels, attFeature, device)
                attTpLoss = criterionTriplet(anchor, positive, negative)

                patchImg = imgSplit(img=inputImgs, batch=inputImgs.size(dim=0))
                avgGriFeature = distanceAverage(saveFeature.returnFeature()[0][0], 384, inputImgs.size(dim=0))
                avgattFeature = distanceAverage(saveFeature.returnFeature()[1][0], 384, inputImgs.size(dim=0))
                inputFeature = torch.cat([avgGriFeature, avgattFeature, saveFeature.returnFeature()[2][0], patchImg], dim=1)

                ##GCELoss
                GCENet.multiFeature = True
                gceOutputs, gceFeature, gceLayerFeature = GCENet(inputFeature)
                anchor, positive, negative = choiceData(labels, gceFeature, device)
                tpLoss = criterionTriplet(anchor, positive, negative)

                if distillFlag is True:
                    # logit distill
                    oriLogitDistill = criterion(oriLayerFeature[3], teacherOriFeature[3])
                    attLogitDistill = criterion(attLayerFeature[3], teacherAttFeature[3])
                    gceLogitDistill = criterion(gceLayerFeature[2], teacherGCEFeature[2])
                    logitDistill = oriLogitDistill + attLogitDistill + gceLogitDistill

                    convNeXtParam = optimalParam.ConvNeXtOptimizing(teacherOriFeature, teacherAttFeature)
                    owkdLoss = OWKDLoss.Loss(oriLayerFeature, attLayerFeature, teacherOriFeature,
                                               teacherAttFeature,
                                               inputFeature, gceLayerFeature, teacherGCEFeature, convNeXtParam)

                    distillLoss = 1e-4 * owkdLoss + 1e-2*logitDistill

                    gceLoss = tpLoss + oriTpLoss + attTpLoss
                    totalLoss = gceLoss + distillLoss
                    ## gceInformation
                else:
                    gceLoss = tpLoss + oriTpLoss + attTpLoss
                    totalLoss = gceLoss

                for labelCnt in range(gceOutputs.shape[0]):
                    label = int(gallerySaveLabels[labelCnt])

                    ReIDkey = label

                    if keyTmp != ReIDkey:
                        value = ReIDdict.get(keyTmp)
                        if value != None:
                            for valueCnt in range(len(ReIDresult)):
                                value.append(ReIDresult[valueCnt])
                            ReIDdict[keyTmp] = value
                        else:
                            ReIDdict[keyTmp] = ReIDresult

                        ReIDresult = []
                    ReIDresult.append([gceFeature[labelCnt], totalLoss.item(), ReIDkey])
                    keyTmp = ReIDkey

                pbar.set_postfix(loss_total=totalLoss.item())
            pbar.close()

            value = ReIDdict.get(keyTmp)
            if value != None:
                for valueCnt in range(len(ReIDresult)):
                    value.append(ReIDresult[valueCnt])
                ReIDdict[keyTmp] = value
            else:
                ReIDdict[keyTmp] = ReIDresult

        saveValResultName = gceModelPath.split('.')[0]
        valResultPath = './' + args.inputDB + '\\' + 'valResult/' + 'gceNet' + '\\' + Fold + '\\' + '/epochTermValidation/'
        if not os.path.isdir(valResultPath):
            os.makedirs(valResultPath)
        with open(valResultPath + 'ReID_val_result' + '_' + saveValResultName + '.pickle',
                  'wb') as fw:
            pickle.dump(ReIDdict, fw)

    del valSet
    del loader
    torch.cuda.empty_cache()
