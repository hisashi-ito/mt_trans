#! /usr/local/bin/python2.7
# -*- coding: utf-8 -*-
#
#【encoder_decoder】
#
# 概要: minibatch に対応した
#       encoder,decoderネットワーク実装
#
import logging
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Function, gradient_check, Variable,optimizers, serializers, utils
from chainer import Link, Chain, ChainList

class EncoderDecoder(chainer.Chain):
	def __init__(self, logging, fv, tv, k, id2voc, voc2id, fvocab, tvocab):
		self.logging = logging
		self.fv = fv          # 翻訳元の総語彙数
		self.tv = tv          # 翻訳先の総語彙数
		self.k  = k           # 隠れ層の次元数
		self.id2voc = id2voc  # idと語彙の対応表
		self.voc2id = voc2id  # 語彙とidの対応表
		self.fvocab  = fvocab
		self.tvocab  = tvocab
		
		# chainer.Chain instance
		super(EncoderDecoder, self).__init__(
			embedx = L.EmbedID(self.fv, self.k, ignore_label=-1),
			embedy = L.EmbedID(self.tv, self.k, ignore_label=-1),
			H = L.LSTM(self.k, self.k),
			W = L.Linear(self.k, self.tv),
			)

	def eos_ary(self,size):
		ary = []
		for i in range(size):
			ary.append(self.voc2id['<eos>'])
		return ary

	def eos_ary2(self,size):
		ary = []
		for i in range(size):
			ary.append(self.tvocab['<eos>'])
		return ary

	# __call__
	# １つのmibatch単位でcallされる
	# 入力はすべてでID化,オリジナルの行列が 転置行列になっているので回すだけでO.K.
	def __call__(self, fbatch, tbatch):
		# encode 
		for i in range(len(fbatch.T)):
			x_k = self.embedx(Variable(np.array(fbatch.T[i],dtype=np.int32)))
			h   = self.H(x_k)
		x_k = self.embedx(Variable(np.array(self.eos_ary(len(fbatch)), dtype=np.int32)))
		h = self.H(x_k)
		tx = Variable(np.array(tbatch.T[0], dtype=np.int32))
		accum_loss = F.softmax_cross_entropy(self.W(h), tx)

		# decode
		for i in range(len(tbatch.T)):
			x_k = self.embedy(Variable(np.array(tbatch.T[i], dtype=np.int32)))
			h = self.H(x_k)
			if i <= len(tbatch.T) - 2: 
				tx = Variable(np.array(tbatch.T[i+1], dtype=np.int32))
			else:
				tx = Variable(np.array(self.eos_ary2(len(tbatch.T[0])), dtype=np.int32))
			loss = F.softmax_cross_entropy(self.W(h), tx)
			accum_loss += loss
		return accum_loss
