import torch
import torch.nn as nn

def accuracy(out, yb):
    softmax = nn.Softmax(dim=1)
    out = softmax(out)
    yb = yb.cpu()

    compare = []
    for i in range(len(out)):
        outList = out[i].tolist()
        tmp = max(outList)
        index = outList.index(tmp)
        compare.append(index)
    compare = torch.Tensor(compare).long()

    return (compare == yb).float().mean()