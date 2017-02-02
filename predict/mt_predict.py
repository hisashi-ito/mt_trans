#! /usr/local/bin/python2.7
# -*- coding: utf-8 -*-
# 
#【mt_predict】
#
# 概要: デコーダ,エンコーダを用いた翻訳ネットワーククラスの
#       推定クラスライブラリ(非GPU対応とする)
# 
# usage: mt_predict.py -m <model-file>
#                      --voc2id <vo2id-file>
#                      --id2voc <id2voc-file>
# 更新履歴:
#           2016.10.22 新規作成
#           2016.10.28 id2voc の <eos> のid 番号を保持する
#
import sys
sys.path.append(".")
import os
import argparse
import logging
import tokenizer as T
import decoder_encoder_predict as ED
from chainer import serializers

class MtPredict:
	""" encoder, decoder 翻訳推定クラス """
	def __init__(self, logging, attention, model_file, voc2id_file, id2voc_file):
		self.logging   = logging
		self.attention = attention
		self.model_file = model_file
		# マッピングファイルの読み込み
		# このファイルには既に <eos> が付与されている
		self.voc2id = self.store_mapping(voc2id_file, "voc2id")
		self.id2voc = self.store_mapping(id2voc_file, "id2voc")
		# eos_id の発見
		self.eos_id = self.read_eos(id2voc_file)
		
		# 深層学習系パラメータ (ハードコーディング)
		# - 推定側と同じになっていないといけない
		self.k = 512
		# encoder, decoder のインスタンス
		self.model = ED.EncoderDecoder(self.logging, len(self.voc2id), 
																	 len(self.id2voc), self.k, self.id2voc, self.voc2id, self.eos_id)
		serializers.load_npz(self.model_file, self.model)
		# 分かち書きインスタンスを作成
		self.to = T.Tokenizer()

	def store_mapping(self, file, kind):
		hash = {}
		with open(file, "r") as f:
			line = f.readline()
			while line:
				elems = line.strip().split("\t")
				if kind == "voc2id":
					hash[str(elems[0])] = int(elems[1])
				else:
					hash[int(elems[0])] = str(elems[1])
				line = f.readline()
		return hash
	
	def read_eos(self, file):
		# id2voc から<eos>のid番号を取得する
		eos_id = None
		with open(file, "r") as f:
			 line = f.readline()
			 while line:
				 elems = line.strip().split("\t")
				 id = int(elems[0])
				 voc = elems[1]
				 if voc == "<eos>":
					 eos_id = id
					 return int(eos_id)
				 line = f.readline()
		return eos_id

	def predict(self, text, lang):
		# 分かち書きを実施
		tokens = self.to.tokenize(text, lang)
		# 反対側にする
		tokensr = tokens[::-1]
		mt_text = self.model.predict(tokensr)
		return " ".join(mt_text)

# メイン関数
def main():
	# logging の初期化
	logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
	parser = argparse.ArgumentParser(description='ENCODE,DECODE 推定プログラム')
	parser.add_argument('-i', action="store", dest="text", default=True)
	parser.add_argument('-m', action="store", dest="model_file", default=True)
	parser.add_argument('--voc2id', action="store", dest="voc2id_file", default=True)
	parser.add_argument('--id2voc', action="store", dest="id2voc_file", default=True)
	parser.add_argument('--attention', action='store_true', dest="attention",default=False)
	options = parser.parse_args()
	logging.info("*** start mt_predict ***")
	mp = MtPredict(logging, options.attention, options.model_file, options.voc2id_file, options.id2voc_file)
	print mp.predict(options.text, "ja")
	logging.info("*** end mt_predict ***")
	
if __name__ == "__main__":
	main()
