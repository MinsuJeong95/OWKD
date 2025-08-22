import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

import pickle
import utils.imgPreprocess as imgPreprocess
import utils.CustomDataset as CustomDataset

import models.convnext as convnext
import models.convnextAttention as convnextAttention
import GCENet.GCENetWithECA as gcenet
import kdLoss.kd as kd

from utils.choiceData import choiceData
from utils.accuracy import accuracy
from utils import featureExtract
from utils.distanceAverage import distanceAverage

import kdLoss.OWKD as OWKD


def training(args, Fold):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.__version__)
    print(device)

    trainDBPath = args.DBPath + '\\' + args.inputDB + '\\' + Fold + '\\train\\allcam'
    numOfClass = len(os.listdir(args.DBPath + '\\' + args.inputDB + '\\' + Fold + '\\train\\allcam'))
    print("numOfClass :", numOfClass)
    print('Fold : ' + Fold)

    GCENet = gcenet.GCENet(inputChannel=384 * 2 + 1 + 1024)
    GCENet.fc = nn.Linear(384 * 2 + 1 + 1024, numOfClass)
    oriModel = convnext.convnext_micro(pretrained=True, num_classes=numOfClass)
    attModel = convnextAttention.convnext_micro(pretrained=True, num_classes=numOfClass)

    pathLen = len(trainDBPath.split('\\'))
    fileName = trainDBPath.split('\\')[pathLen - 4] + '_' + trainDBPath.split('\\')[pathLen - 3] + '_' + \
               trainDBPath.split('\\')[pathLen - 1]

    oriPATH = './' + args.inputDB + '/trainModels/' + 'original/' + \
              Fold + '/epochTermModel/'
    if not os.path.isdir(oriPATH):
        os.makedirs(oriPATH)
    attPATH = './' + args.inputDB + '/trainModels/' + 'attention/' + \
              Fold + '/epochTermModel/'
    if not os.path.isdir(attPATH):
        os.makedirs(attPATH)
    gcePATH = './' + args.inputDB + '/trainModels/' + 'gceNet/' + \
              Fold + '/epochTermModel/'
    if not os.path.isdir(gcePATH):
        os.makedirs(gcePATH)
    lossModelPATH = './' + args.inputDB + '/trainModels/' + 'lossModel/' + \
                    Fold + '/epochTermModel/'
    if not os.path.isdir(lossModelPATH):
        os.makedirs(lossModelPATH)

    trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    trainSet = CustomDataset.CustomDataset(root_dir=trainDBPath, transforms=trans)
    loader = DataLoader(trainSet, batch_size=args.trainBatchSize, shuffle=True, pin_memory=True)

    oriModel.train()
    oriModel.to(device)
    attModel.train()
    attModel.to(device)
    GCENet.train()
    GCENet.to(device)

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

    criterion = nn.CrossEntropyLoss().to(device)
    criterionTriplet = nn.TripletMarginLoss().to(device)

    OWKDLoss = OWKD.OWKD_loss(device)
    optimalParam = kd.kdlossOptimizing().to(device)

    GCENetOptimizer = torch.optim.Adam([
        {'params': oriModel.parameters(), 'lr': args.lr, 'weight_decay': args.wd},
        {'params': attModel.parameters(), 'lr': args.lr, 'weight_decay': args.wd},
        {'params': GCENet.parameters(), 'lr': args.GCE_lr, 'weight_decay': args.GCE_wd},
        {'params': optimalParam.parameters(), 'lr': args.lr, 'weight_decay': args.wd}]
        )

    saveFeature = featureExtract.saveFeature()
    oriModel.stages[-1].register_forward_hook(saveFeature.saveOriFeature)
    attModel.stages[-1].register_forward_hook(saveFeature.saveAttFeature)
    attModel.stages[-1][1].SpatialGate.spaSig.register_forward_hook(saveFeature.saveCBAMSpatialFeature)
    imgSplit = imgPreprocess.imgPreprocess().imgSplit

    saveOriIterCnt = []
    saveOriLoss = []
    saveOriAccuracy = []
    saveAttIterCnt = []
    saveAttLoss = []
    saveAttAccuracy = []
    saveGceIterCnt = []
    saveGceLoss = []
    saveTotalLoss = []
    saveGceAccuracy = []
    iterCnt = 0
    distillFlag = True
    logitDistill = torch.zeros(1)
    owkdLoss = torch.zeros(1)

    for epoch in range(args.numEpoch):
        loaderIter = iter(loader)
        pbar = tqdm(range(len(loader)))
        pbar.set_description(f'Epoch - {epoch}')
        for pbar_i in pbar:
            Data = next(loaderIter)
            inputImgs = Data['image'].to(device)
            labels = Data['label'].to(device)

            teacherOriFeature, teacherAttFeature, teacherGCEFeature, teacherOriOutPuts, teacherAttOutPuts, teacherGCEOutPuts = \
                kd.extractTeacherFeature(teacherOri, teacherAtt, teacherGCE, inputImgs)

            ## GCE
            saveFeature.initFeature()
            oriModel.wsFlag = True
            oriModel.multiFeature = False
            oriOutputs, oriWSoutputs, oriFeature, oriLayerFeature = oriModel(inputImgs)
            ceLoss = criterion(oriOutputs, labels) + criterion(oriWSoutputs, labels)
            anchor, positive, negative = choiceData(labels, oriFeature, device)
            tpLoss = criterionTriplet(anchor, positive, negative)
            oriLoss = ceLoss + tpLoss
            oriAcc = accuracy(oriOutputs, labels)
            saveOriAccuracy.append(oriAcc)

            attModel.wsFlag = True
            attModel.multiFeature = False
            attOutputs, attWSoutputs, attFeature, attLayerFeature = attModel(inputImgs)
            ceLoss = criterion(attOutputs, labels) + criterion(attWSoutputs, labels)
            anchor, positive, negative = choiceData(labels, attFeature, device)
            tpLoss = criterionTriplet(anchor, positive, negative)
            attLoss = ceLoss + tpLoss
            attAcc = accuracy(attOutputs, labels)
            saveAttAccuracy.append(attAcc)

            patchImg = imgSplit(img=inputImgs, batch=inputImgs.size(dim=0))
            avgOriFeature = distanceAverage(saveFeature.returnFeature()[0][0], 384, inputImgs.size(dim=0))
            avgAttFeature = distanceAverage(saveFeature.returnFeature()[1][0], 384, inputImgs.size(dim=0))
            inputFeature = torch.cat([avgOriFeature, avgAttFeature, saveFeature.returnFeature()[2][0], patchImg], dim=1)

            GCENet.multiFeature = False
            gceOutputs, gceFeature, gceLayerFeature = GCENet(inputFeature)
            ##GCELoss
            GCENetOptimizer.zero_grad()
            ceLoss = criterion(gceOutputs, labels)
            anchor, positive, negative = choiceData(labels, gceFeature, device)
            tpLoss = criterionTriplet(anchor, positive, negative)
            gceLoss = ceLoss + tpLoss

            if distillFlag is True:
                # logit distill
                oriLogitDistill = criterion(oriOutputs, teacherOriOutPuts.softmax(dim=1).detach())
                attLogitDistill = criterion(attOutputs, teacherAttOutPuts.softmax(dim=1).detach())
                gceLogitDistill = criterion(gceOutputs, teacherGCEOutPuts.softmax(dim=1).detach())
                logitDistill = oriLogitDistill + attLogitDistill + gceLogitDistill

                convNeXtParam = optimalParam.ConvNeXtOptimizing(teacherOriFeature, teacherAttFeature)
                owkdLoss = OWKDLoss.Loss(oriLayerFeature, attLayerFeature, teacherOriFeature, teacherAttFeature,
                                      inputFeature, gceLayerFeature, teacherGCEFeature, convNeXtParam)

                distillLoss = 1e-4 * owkdLoss + 1e-2 * logitDistill
                totalLoss = oriLoss + attLoss + gceLoss + distillLoss
            else:
                distillLoss = torch.zeros(1)
                totalLoss = gceLoss + oriLoss + attLoss

            totalLoss.backward()
            GCENetOptimizer.step()
            # gceInformation
            gceAcc = accuracy(gceOutputs, labels)
            saveGceIterCnt.append(iterCnt)
            saveGceLoss.append(gceLoss.item())
            saveOriLoss.append(oriLoss.item())
            saveAttLoss.append(attLoss.item())
            saveTotalLoss.append(totalLoss.item())
            saveGceAccuracy.append(gceAcc)

            pbar.set_postfix(loss_ori=oriLoss.item(), loss_att=attLoss.item(), loss_gce=gceLoss.item(),
                             loss_logitDistill=logitDistill.item(),
                             loss_OWKD=owkdLoss.item(),
                             loss_DistillTotal=distillLoss.item(),
                             loss_Total=totalLoss.item(),
                             acc_ori=float(oriAcc), acc_att=float(attAcc), acc_gce=float(gceAcc), )

            iterCnt = iterCnt + 1

        lastEpoch = epoch + 1

        # original model info save
        trainPath = oriPATH + fileName + '_ReID_' + 'original' + '_' + \
                    str(lastEpoch) + '.pth'
        torch.save(oriModel.state_dict(), trainPath)
        oriTrainInfoPath = './' + args.inputDB + '/trainModels/' + 'original' + '/' + Fold \
                           + '/saveEpochInfo/'
        if not os.path.isdir(oriTrainInfoPath):
            os.makedirs(oriTrainInfoPath)
        with open(oriTrainInfoPath + fileName + '_ReID_' + 'original' + '_saveIterCnt_' + str(lastEpoch) + '.pickle',
                  'wb') as fw:
            pickle.dump(saveOriIterCnt, fw)
        with open(oriTrainInfoPath + fileName + '_ReID_' + 'original' + '_saveLoss_' + str(lastEpoch) + '.pickle',
                  'wb') as fw:
            pickle.dump(saveOriLoss, fw)
        with open(oriTrainInfoPath + fileName + '_ReID_' + 'original' + '_saveAccuracy_' + str(lastEpoch) + '.pickle',
                  'wb') as fw:
            pickle.dump(saveOriAccuracy, fw)

        # attention model info save
        trainPath = attPATH + fileName + '_ReID_' + 'attention' + '_' + \
                    str(epoch + 1) + '.pth'
        torch.save(attModel.state_dict(), trainPath)
        attTrainInfoPath = './' + args.inputDB + '/trainModels/' + 'attention' + '/' + Fold \
                           + '/saveEpochInfo/'
        if not os.path.isdir(attTrainInfoPath):
            os.makedirs(attTrainInfoPath)
        with open(attTrainInfoPath + fileName + '_ReID_' + 'attention' + '_saveIterCnt_' + str(lastEpoch) + '.pickle',
                  'wb') as fw:
            pickle.dump(saveAttIterCnt, fw)
        with open(attTrainInfoPath + fileName + '_ReID_' + 'attention' + '_saveLoss_' + str(lastEpoch) + '.pickle',
                  'wb') as fw:
            pickle.dump(saveAttLoss, fw)
        with open(attTrainInfoPath + fileName + '_ReID_' + 'attention' + '_saveAccuracy_' + str(lastEpoch) + '.pickle',
                  'wb') as fw:
            pickle.dump(saveAttAccuracy, fw)

        # gce-Net info save
        trainPath = gcePATH + fileName + '_ReID_' + 'gceNet' + '_' + \
                    str(epoch + 1) + '.pth'
        torch.save(GCENet.state_dict(), trainPath)
        gceTrainInfoPath = './' + args.inputDB + '/trainModels/' + 'gceNet' + '/' + Fold \
                           + '/saveEpochInfo/'
        if not os.path.isdir(gceTrainInfoPath):
            os.makedirs(gceTrainInfoPath)
        with open(gceTrainInfoPath + fileName + '_ReID_' + 'gceNet' + '_saveIterCnt_' + str(lastEpoch) + '.pickle',
                  'wb') as fw:
            pickle.dump(saveGceIterCnt, fw)
        with open(gceTrainInfoPath + fileName + '_ReID_' + 'gceNet' + '_saveLoss_' + str(lastEpoch) + '.pickle',
                  'wb') as fw:
            pickle.dump(saveGceLoss, fw)
        with open(gceTrainInfoPath + fileName + '_ReID_' + 'gceNet' + '_saveAccuracy_' + str(lastEpoch) + '.pickle',
                  'wb') as fw:
            pickle.dump(saveGceAccuracy, fw)

        # total loss save
        trainPath = lossModelPATH + fileName + '_ReID_' + 'lossOptimizer' + '_' + \
                    str(epoch + 1) + '.pth'
        torch.save(optimalParam.state_dict(), trainPath)
        totalLossInfoPath = './' + args.inputDB + '/trainModels/' + 'totalLoss' + '/' + Fold \
                            + '/saveEpochInfo/'
        if not os.path.isdir(totalLossInfoPath):
            os.makedirs(totalLossInfoPath)
        with open(totalLossInfoPath + fileName + '_ReID_' + 'wseNet' + '_saveIterCnt_' + str(lastEpoch) + '.pickle',
                  'wb') as fw:
            pickle.dump(saveGceIterCnt, fw)
        with open(totalLossInfoPath + fileName + '_ReID_' + 'wseNet' + '_saveTotalLoss_' + str(lastEpoch) + '.pickle',
                  'wb') as fw:
            pickle.dump(saveTotalLoss, fw)

    del trainSet
    del loader
    torch.cuda.empty_cache()

    print('Finished Training')
