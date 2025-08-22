import torch


def distanceAverage(gradient, channel, batch):
    spGrad = gradient.split(32, dim=1)
    spGrad = torch.stack(spGrad, dim=1)
    spGrad = spGrad.split(int(768/channel), dim=2)
    spGrad = torch.stack(spGrad, dim=1)
    gradResult = spGrad.mean(dim=3)

    gradResult = torch.reshape(gradResult, (batch, -1, 7, 7))
    return gradResult