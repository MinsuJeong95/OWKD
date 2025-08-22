def sortModelList(modelPaths):
    modelNum = []
    # 모델list 재배열
    for i in range(len(modelPaths)):
        modelName = modelPaths[i].split('_')
        modelNum.append(int(modelName[len(modelName) - 1].split('.')[0]))
    modelNum.sort()

    # 재배열 적용
    for num_i in range(len(modelNum)):
        modelName = modelPaths[num_i].split('_')
        del modelName[len(modelName) - 1]
        modelName.append(str(modelNum[num_i]) + '.pth')
        saveName = ''
        # 나눠져 있는 list 재 연결
        for name_i in range(len(modelName)):
            if name_i == (len(modelName) - 1):
                saveName = saveName + modelName[name_i]
            else:
                saveName = saveName + modelName[name_i] + '_'
        modelPaths[num_i] = saveName

    print(len(modelPaths))

    return modelPaths
