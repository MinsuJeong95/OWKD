from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.layerTransform import Sample
import numpy as np


class FSP(nn.Module):
	'''
	A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning
	http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf
	'''
	def __init__(self):
		super(FSP, self).__init__()

	def forward(self, fm_s1, fm_s2, fm_t1, fm_t2):
		loss = F.mse_loss(self.fsp_matrix(fm_s1,fm_s2), self.fsp_matrix(fm_t1,fm_t2))

		return loss

	def fsp_matrix(self, fm1, fm2):
		if fm1.size(2) > fm2.size(2):
			fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))

		fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
		fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1,2)

		fsp = torch.bmm(fm1, fm2) / fm1.size(2)

		return fsp


class FSP_with_ft(nn.Module):
	'''
	A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning
	http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf
	'''
	def __init__(self):
		super(FSP_with_ft, self).__init__()

	def forward(self, fm_s1, fm_s2, fm_t1, fm_t2):
		fts, _ = self.featureFT(fm_s1, fm_s2)
		ftt, _ = self.featureFT(fm_t1, fm_t2)
		loss = F.mse_loss(fts, ftt)

		return loss

	def fsp_matrix(self, fm1, fm2):
		if fm1.size(2) > fm2.size(2):
			fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))

		fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
		fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1,2)

		fsp = torch.bmm(fm1, fm2) / fm1.size(2)

		return fsp

	def featureFT(self, fm_1, fm_2):
		matrix = self.fsp_matrix(fm_1, fm_2)
		ft = torch.abs(torch.fft.fft2(matrix, norm='ortho'))
		return ft, matrix


class FSP_with_ft_ver2(nn.Module):
	'''
	A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning
	http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf
	'''
	def __init__(self):
		super(FSP_with_ft_ver2, self).__init__()

	def forward(self, fm_s1, fm_s2, fm_t1, fm_t2):
		fts, gms = self.featureFT(fm_s1, fm_s2)
		ftt, gmt = self.featureFT(fm_t1, fm_t2)
		loss = F.mse_loss(fts, ftt) + F.mse_loss(gms, gmt)

		return loss

	def fsp_matrix(self, fm1, fm2):
		if fm1.size(2) > fm2.size(2):
			fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))

		fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
		fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1,2)

		fsp = torch.bmm(fm1, fm2) / fm1.size(2)

		return fsp

	def featureFT(self, fm_1, fm_2):
		matrix = self.fsp_matrix(fm_1, fm_2)
		ft = torch.abs(torch.fft.fft2(matrix, norm='ortho'))
		return ft, matrix


class FSP_with_ft_ver3(nn.Module):
	'''
	A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning
	http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf
	'''
	def __init__(self):
		super(FSP_with_ft_ver3, self).__init__()

	def forward(self, fm_s1, fm_s2, fm_t1, fm_t2, optimalParam):
		fts, gms = self.featureFT(fm_s1, fm_s2)
		ftt, gmt = self.featureFT(fm_t1, fm_t2)
		# param = optimalParam.unsqueeze(dim=1).unsqueeze(dim=1)
		# loss = F.mse_loss(fts * param, ftt * param) + F.mse_loss(gms * param, gmt * param)
		loss = 0
		for i in range(optimalParam.size(0)):
			loss = loss + optimalParam[i] * (F.mse_loss(fts[i], ftt[i]) + F.mse_loss(gms[i], gmt[i]))

		return loss, fts

	def fsp_matrix(self, fm1, fm2):
		if fm1.size(2) > fm2.size(2):
			fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))

		fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
		fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1,2)

		fsp = torch.bmm(fm1, fm2) / fm1.size(2)

		return fsp

	def featureFT(self, fm_1, fm_2):
		matrix = self.fsp_matrix(fm_1, fm_2)
		ft = torch.abs(torch.fft.fft2(matrix, norm='ortho'))
		return ft, matrix


class OWKD_loss:
	def __init__(self, device):
		super(OWKD_loss, self).__init__()
		self.FSPft_ver2 = FSP_with_ft_ver2().to(device)
		self.FSPft_ver3 = FSP_with_ft_ver3().to(device)

	def Loss(self, oriLayerFeature, attLayerFeature, teacherOriFeature, teacherAttFeature, inputFeature,
			 gceLayerFeature, teacherGCEFeature, optimalParam):
		FSPft_ver3loss = 0

		lossOptimalParam = optimalParam

		for fsp_i in range(4):
			FSPft_ver3loss = FSPft_ver3loss + self.FSPft_ver3(oriLayerFeature[fsp_i], attLayerFeature[fsp_i],
															  teacherOriFeature[fsp_i], teacherAttFeature[fsp_i],
															  lossOptimalParam[fsp_i])[0]

		gceKnowledgeDistillft = 0
		for fsp_i in range(3):
			if fsp_i == 0:
				gceKnowledgeDistillft = self.FSPft_ver2(inputFeature, gceLayerFeature[0], inputFeature, teacherGCEFeature[0])
			else:
				gceKnowledgeDistillft = gceKnowledgeDistillft + self.FSPft_ver2(gceLayerFeature[fsp_i-1], gceLayerFeature[fsp_i],
																				teacherGCEFeature[fsp_i-1], teacherGCEFeature[fsp_i])
		loss = FSPft_ver3loss + gceKnowledgeDistillft

		return loss
