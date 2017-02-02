#! /usr/local/bin/python2.7
# -*- coding: utf-8 -*-
#
#【decoder_encoder_predict】
#
# 概要: デコーダ,エンコーダを用いた翻訳推定クラス [CPU対応版]
#
# 更新履歴: 
#           2016.10.25 新規実装
#
import logging
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Function, gradient_check, Variable,optimizers, serializers, utils
from chainer import Link, Chain, ChainList

class EncoderDecoder(chainer.Chain):
	def __init__(self, logging, fv, tv, k, id2voc, voc2id, eos_id):
		# インスタンス変数の初期化
		self.logging = logging
		self.fv = fv
		self.tv = tv
		self.k = k
		self.id2voc = id2voc
		self.voc2id = voc2id
		# decoder側の eos のID番号
		self.eos_id = int(eos_id)

		# 打ち切り語数
		self.terminate_num = 30
		
		# chainer.Chain インスタンスの初期化
		super(EncoderDecoder, self).__init__(
			embedx = L.EmbedID(self.fv, self.k),
			embedy = L.EmbedID(self.tv, self.k),
			H = L.LSTM(self.k, self.k),
			W = L.Linear(self.k, self.tv),
			)

	# 分かち書きされた文字列(tokens)をとり推定処理を実施する
	def predict(self, tokens):
		# 入力文の解析(encode 処理)
		for w in tokens:
			if self.voc2id.has_key(w):
				# voc -> id 
				id = self.voc2id[w]
				x_k = self.embedx(Variable(np.array([id], dtype=np.int32),volatile='on'))
				h = self.H(x_k)
		# 翻訳元文の<eos>の処理
		id = self.voc2id['<eos>']
		x_k = self.embedx(Variable(np.array([id], dtype=np.int32),volatile='on'))
		h = self.H(x_k)
	  
		# 出力文の解析(decode 処理) ・・・ argmax は最大の要素(index)を指定
		id = np.argmax(F.softmax(self.W(h)).data[0])
		
		# 翻訳後の１文字目を保存
		retval = []
		if id == self.eos_id:
			return retval
		if self.id2voc.has_key(id) and (id != self.eos_id):
			retval.append(self.id2voc[id])
		loop = 0
		while (id != self.eos_id) and (loop <= self.terminate_num):
			x_k = self.embedy(Variable(np.array([id], dtype=np.int32), volatile='on'))
			h = self.H(x_k)
			id = np.argmax(F.softmax(self.W(h)).data[0])
			if id == self.eos_id:
				return retval
			if self.id2voc.has_key(id) and  id != self.eos_id:
				retval.append(self.id2voc[id])
			loop += 1
		return retval
