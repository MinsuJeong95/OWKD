import torch

def findTopFeature(features):
    distenceResult = []
    resultImg = 0

    # Calculate Images Center
    for center in range(len(features)):
        resultAll = 0
        for f_i in range(len(features)):
            if center == f_i:
                continue
            result = torch.dist(features[center], features[f_i])
            resultAll += result
        resultAll /= (len(features)-1)

        distenceResult.append(resultAll)
    pickImg = min(distenceResult)
    for d_i in range(len(distenceResult)):
        if pickImg == distenceResult[d_i]:
            resultImg = features[d_i]
            break

    return resultImg