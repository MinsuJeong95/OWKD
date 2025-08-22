import torch
import torch.nn as nn

from utils import featureExtract
from utils import imgPreprocess
from utils.distanceAverage import distanceAverage


def extractTeacherFeature(oriModel, attModel, GCENet, inputImgs):
    handler = []
    saveFeature = featureExtract.saveFeature()
    handler.append(oriModel.stages[-1].register_forward_hook(saveFeature.saveOriFeature))
    handler.append(attModel.stages[-1].register_forward_hook(saveFeature.saveAttFeature))
    handler.append(attModel.stages[-1][3].SpatialGate.spaSig.register_forward_hook(saveFeature.saveCBAMSpatialFeature))
    imgSplit = imgPreprocess.imgPreprocess().imgSplit

    oriModel.wsFlag = False
    oriModel.multiFeature = False
    attModel.wsFlag = False
    attModel.multiFeature = False
    saveFeature.initFeature()
    oriOutputs, _, oriFeature = oriModel(inputImgs)
    attOutputs, _, attFeature = attModel(inputImgs)
    patchImg = imgSplit(img=inputImgs, batch=inputImgs.size(dim=0))
    oriInputFeature = distanceAverage(saveFeature.returnFeature()[0][0], 384, inputImgs.size(dim=0))
    attInputFeature = distanceAverage(saveFeature.returnFeature()[1][0], 384, inputImgs.size(dim=0))
    inputFeature = torch.cat([oriInputFeature, attInputFeature, saveFeature.returnFeature()[2][0], patchImg], dim=1)

    GCENet.multiFeature = False
    gceOutputs, _, gceLayerFeature = GCENet(inputFeature)

    handler[0].remove()
    handler[1].remove()
    handler[2].remove()
    return oriFeature, attFeature, gceLayerFeature, oriOutputs, attOutputs, gceOutputs


def loss(studentFeatures, teacherFeatures, criterion):
    result = 0
    for i in range(len(teacherFeatures)):
        result = result + criterion(studentFeatures[i], teacherFeatures[i].detach())
    return result


def multiFeatureLoss(studentFeatures, teacherFeatures, criterion):
    result = 0
    for i, studentFeature in enumerate(studentFeatures):
        if i % 5 == 0:
            loss = criterion(studentFeature, teacherFeatures[i % len(teacherFeatures)].detach())
        else:
            loss = 0.1 * criterion(studentFeature, teacherFeatures[i % len(teacherFeatures)].detach())
        result = result + loss
    return result


def multiFeatureLoss_gce(studentFeatures, teacherFeatures, criterion):
    result = 0
    for i, studentFeature in enumerate(studentFeatures):
        if i % 4 == 0:
            loss = criterion(studentFeature, teacherFeatures[i % len(teacherFeatures)].detach())
        else:
            loss = 0.1 * criterion(studentFeature, teacherFeatures[i % len(teacherFeatures)].detach())
        result = result + loss
    return result


class kdlossOptimizing(nn.Module):
    def __init__(self):
        super(kdlossOptimizing, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # self.hiddenFC1 = nn.Linear(1536, 768)
        # self.hiddenFC2 = nn.Linear(768, 256)
        self.convnextFC = nn.Linear(1536, 4)
        self.softmax = nn.Softmax(dim=1)

    def ConvNeXtOptimizing(self, oriFeature, attFeature):
        poolingOriFeature = self.gap(oriFeature[3]).view(-1, 768)
        poolingAttFeature = self.gap(attFeature[3]).view(-1, 768)
        totalFeature = torch.cat((poolingOriFeature, poolingAttFeature), 1)
        # optimalParam = self.hiddenFC1(totalFeature)
        # optimalParam = self.hiddenFC2(optimalParam)
        optimalParam = self.convnextFC(totalFeature)
        optimalParam = len(optimalParam[1])*self.softmax(optimalParam).transpose(0, 1)

        return optimalParam






