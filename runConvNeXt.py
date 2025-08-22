import os
import Training
import Validation
import validationAcc
import Test
import calculate.APCalculate as APCalculate
import calculate.RankCalculate as RankCalculate


def run(args, models):
    for datasetType in args.datasetTypes:
        for model_i, modelType in enumerate(args.modelTypes):
            for Fold in args.Folds:
                trainDBPath = args.DBPath + '\\' + datasetType + '\\' + Fold + '\\train\\allcam'
                valDBPath = args.DBPath + '\\' + datasetType + '\\' + Fold + '\\validation\\allcam'
                testDBPath = args.DBPath + '\\' + datasetType + '\\' + Fold + '\\test\\allcam'

                # numOfClass = len(os.listdir(trainDBPath))

                Training.training(args, Fold, modelType, trainDBPath, models[model_i], datasetType)
                # Validation.validation(datasetType, modelType, Fold, models[model_i], valDBPath, numOfClass)
                # validationAcc.accCalculate(datasetType, modelType, Fold)
                # Test.test(datasetType, modelType, Fold, models[model_i], testDBPath, numOfClass)
                # APCalculate.apCalculate(datasetType, modelType, Fold)
                # RankCalculate.rankClaculate(datasetType, modelType, Fold)
