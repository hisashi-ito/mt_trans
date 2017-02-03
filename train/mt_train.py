#! /usr/local/bin/python2.7
# -*- coding: utf-8 -*-
#
#【mt_train.py】
# 
# 概要: デコーダ,エンコーダを用いた翻訳ネットワーク
#       クラスの学習コマンド。ミニバッチの実装版
#
# usage: mt_train.py -i <対訳コーパス>
#                    -d <翻訳方向>
#                    -l <言語方向>
#                    -o <出力ファイルパス>
#
#
# 注意点) 対訳コーパスは以下の条件を満たしたものを入力とする
#         既に分かち書きされており、<sp>で分割されている、
#         1:1 の文対応であるとする
#
# 更新履歴: 2016.11.01 ミニバッチ実装
# 
import sys
import os
import argparse
import logging
import copy
sys.path.append(".")
import decoder_encoder as ED
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Function, gradient_check, Variable,optimizers, serializers, utils
from chainer import Link, Chain, ChainList

class MtTrain:
	def __init__(self, logging, input, output, direction, lang, k , epoch_num, batchsize, step):
		self.logging = logging
		self.input   = input
		self.output  = output
		self.direction = direction
		self.lang = lang

		# ネットワーク設定
		self.k = int(k)                 # 隠れ総の次元数
		self.epoch_num = int(epoch_num) # epoch 数
		self.batchsize = int(batchsize) # バッチサイズ ex) 32
		self.step = int(step)           # モデルの出力ステップ
		
		# 入力文をID配列に変換した変数
		self.ftexts = []
		self.ttexts = []
		
		# fvocab は voc2id と同じになる
		self.fvocab = {}
		self.tvocab = {}
		
		# 総語彙数
		self.fv = 0 # 翻訳元
		self.tv = 0 # 翻訳先
		
		# id,voc の マッピング表
		self.voc2id = {} # fvocab の別名
		self.id2voc = {} # こちらは作成する必要あり

		# コーパスの読み込み
		self.logging.info("start read corpus")
		self.read_corpus()
		self.logging.info("finish read corpus")
		self.logging.info("number of from text: %d",len(self.ftexts))
		self.logging.info("number of to   text: %d",len(self.ttexts))
		self.logging.info("number of voc(from) number: %d",self.fv)
		self.logging.info("number of voc(to)   number: %d",self.tv)
		self.logging.info("mapping file を出力します")
		self.write_mapping(self.output, "voc2id")
		self.write_mapping(self.output, "id2voc")
		self.logging.info("mapping file の出力が完了しました")
		# コーパス数
		self.N = len(self.ftexts)
		
	def minibatch(self, texts, ary):
		retval = []
		for i in ary:
			retval.append(texts[i])
		return retval

	def train(self):
		self.logging.info("学習(train)を開始します")
		model = ED.EncoderDecoder(self.logging, self.fv, self.tv, self.k, self.id2voc, self.voc2id, self.fvocab, self.tvocab)
		optimizer = optimizers.Adam()
		optimizer.setup(model)
		
		# epoch loop
		for epoch in range(self.epoch_num):
			# minibatch loop
			perm = np.random.permutation(self.N)
			for i in range(0, self.N, self.batchsize):
				fbatch = self.minibatch(self.ftexts, perm[i:i + self.batchsize])
				tbatch = self.minibatch(self.ttexts, perm[i:i + self.batchsize])
				# "-1" でパデング処理
				fbatch = self.padding(fbatch)
				tbatch = self.padding(tbatch)
				fbatch = np.array(fbatch, dtype=np.int32)
				tbatch = np.array(tbatch, dtype=np.int32)
				model.H.reset_state()
				model.zerograds()
				loss = model.__call__(fbatch, tbatch)
				loss.backward()
				loss.unchain_backward()
				optimizer.update()
				
			self.logging.info("epoch loop %d: loss %f",epoch, loss.data);
			# 特定ステップでモデルの書き出し
			if int(epoch) % int(self.step) == 0:
				output = self.output + "/mt_" + str(self.lang) +"_" + str(epoch) + ".model"
				serializers.save_npz(output, model)

	# minibatch 計算をするときに(-1)で埋める
	def padding(self, batch):
		max_size = 0
		# 最大文字列長を調査
		for i in range(len(batch)):
			if len(batch[i]) > max_size:
				max_size = len(batch[i])
		for i in range(len(batch)):
			for j in range(max_size):
				if j >= len(batch[i]):
					batch[i].append(-1)
		return batch

	def read_corpus(self):
		cnt1 = 0;
		cnt2 = 0;
		with open(self.input, "r") as file:
			line = file.readline()
			while line:
				elems = line.strip().split("\t")
				if self.direction == "f":
					ftext = elems[0]
					ttext = elems[1]
				else:
					ttext = elems[0]
					ftext = elems[1]
					
				if ftext == "" or ttext == "":
					line = file.readline()
					continue

				# 翻訳元のマップ情報作成
				buf = []
				for w in ftext.split():
					# w のIDを求める or 決定する
					if self.fvocab.has_key(w):
						id = self.fvocab[w]
					else:
						self.fvocab[w] = cnt1
						cnt1 += 1
					# id 番号をbuf につめる
					buf.append(self.fvocab[w])
					# 入力文の出現順を"逆順"にして保存
				self.ftexts.append(copy.deepcopy(buf[::-1]))
				
				# 翻訳先のマップ情報作成
				buf = []
				for w in ttext.split():
					if self.tvocab.has_key(w):
						id = self.tvocab[w]
					else:
						self.tvocab[w]    = cnt2
						self.id2voc[cnt2] = w
						cnt2 += 1
					# id 番号をbuf につめる
					buf.append(self.tvocab[w])
				# 翻訳文をそのまま配列に入れる
				self.ttexts.append(copy.deepcopy(buf))
				line = file.readline()

			# 最後に<eos>を通し番号に付与
			self.fvocab["<eos>"] = cnt1
			self.voc2id = copy.deepcopy(self.fvocab)
			self.tvocab["<eos>"] = cnt2
			self.id2voc[cnt2] = "<eos>"
			
			# 総語彙数の計算
			self.fv = len(self.fvocab)
			self.tv = len(self.tvocab)

	# マッピングファイルの出力
	def write_mapping(self, file_path, kind):
		# ファイル名を種別(kind)から決定
		if kind == "voc2id":
			# 翻訳元言語が対象(from)
			file_name = file_path + "/mt_voc2id.tsv"
		else:
			# 翻訳先言語が対象(to)
			file_name = file_path + "/mt_id2voc.tsv"
		f = open(file_name, "w")
		# kind 毎に mapping ファイルを tsv で書き出す
		if str(kind) == "voc2id":
			for v, i in self.fvocab.iteritems():
				f.write(str(v) + "\t" + str(i) + "\n")
		else:
			for i, v in self.id2voc.iteritems():
				f.write(str(i) + "\t" + str(v) + "\n")
		f.close()

# メイン関数
def main():
	# logging の初期化
	logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
	parser = argparse.ArgumentParser(description='ENCODE,DECODE 翻訳プログラム')
	# 学習時オプション
	parser.add_argument('-i', action="store", dest="input", default=True)
	parser.add_argument('-d', action="store", dest="direction", default=True)
	parser.add_argument('-o', action="store", dest="output", default=True)
	parser.add_argument('-l', action="store", dest="lang", default=True)
	parser.add_argument('-e', action="store", dest="epoch_num", default=True)
	parser.add_argument('-s', action="store", dest="step", default=True)
	parser.add_argument('-m', action="store", dest="batchsize", default=True)
	parser.add_argument('-k', action="store", dest="k", default=True)
	options = parser.parse_args()
	logging.info("*** start mt_train ***")
	mt = MtTrain(logging, options.input, options.output, options.direction,
							 options.lang, options.k, options.epoch_num, options.batchsize, options.step)
	mt.train()
	logging.info("*** end mt_train ***")
	
	
if __name__ == "__main__":
  main()
