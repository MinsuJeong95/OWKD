import numpy as np
import torch

def choiceData(labels, feature, device):
    idxSize = len(labels)
    labels = np.array(labels.to('cpu'))

    if len(set(labels)) != len(labels):
        visited = set()
        dup = [x for x in labels if x in visited or (visited.add(x) or False)]
        sameLabelIdx = np.where(labels == dup[0])[0]

        anchor = feature[sameLabelIdx[0]]
        positive = feature[sameLabelIdx[1]]
        negative = torch.zeros(feature[sameLabelIdx[0]].size()).to(device)

        for i in range(idxSize):
            if i != sameLabelIdx[0] and i != sameLabelIdx[1]:
                negative = feature[i]
                break

    else:
        anchorIdx = int(np.random.choice(idxSize, 1, replace=False))
        anchorLabel = labels[anchorIdx]
        anchor = feature[anchorIdx]
        negative = torch.zeros(feature[anchorIdx].size(0)).to(device)

        for i in range(idxSize):
            if labels[i] != anchorLabel:
                negative = feature[i]
                break

        positive = anchor

    return anchor, positive, negative