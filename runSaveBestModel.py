import os
import saveBestModel.saveBestModel as saveBestModel
import saveBestModel.saveFeatureMap as saveFeatureMap
import saveBestModel.saveFeatureMapWithPatchImage as saveFeatureMapWithPatchImage

def run(args, models):
    saveBestModel.saveBestModel(args.Folds, args.modelTypes, args.datasetTypes)
    for datasetType in args.datasetTypes:
        for Fold in args.Folds:
            trainDBPath = args.DBPath + '\\' + datasetType + '\\' + Fold + '\\train\\allcam'
            numOfClass = len(os.listdir(trainDBPath))
            # Save SCE-Net Training Dataset
            if os.path.isdir('.\\' + datasetType + '\\GCENetTrainData\\' + Fold + '\\train\\'):
                continue
            saveFeatureMapWithPatchImage.saveFeatureMap(datasetType, args.modelTypes, Fold, models, args.DBPath, numOfClass)
            # saveFeatureMap.saveFeatureMap(datasetType, modelTypes, Fold, models, DBPath, numOfClass)