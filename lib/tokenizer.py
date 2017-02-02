#! /usr/local/bin/python2.7
# -*- coding: utf-8 -*-
#
#【形態素解析クラス】
#
# 概要: 形態素解析を日英で実施する
#
import MeCab
import re

class Tokenizer:
	def __init__(self):
		# neologd の辞書の登録
		self.m = MeCab.Tagger(' -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
			
	def tokenize(self, text, lang):
		retval = []
		if lang == "ja":
			retval = self.tokenize_ja(text)
		else:
			retval = self.tokenize_en(text)
		return retval

	def tokenize_ja(self, text):
		retval = []
		pat = r'^BOS'
		res = self.m.parseToNode(text)
		while res:
			form = res.surface
			if form == "":
				res = res.next
				continue
			m = re.match(pat, form)
			if m:
				res = res.next
				continue
			retval.append(form)
			res = res.next
		return retval
	
	def tokenize_en(self, text):
		pat = r"(.+)(^\s).$|(.+)(^\s)!$'|(.+)(^\s)?$"
		m = re.match(pat , text)
		if m:
			# 正規表現に逸しした場合置換する
			text = re.sub(r'\.', ' .', text)
			text = re.sub(r'\!', ' !', text)
			text = re.sub(r'\?', ' ?', text)
		return text.split()



# debug
if __name__ == "__main__":
	t = Tokenizer()
	jtext = "私はバルト９へ行きます。"
	etext = "this is my pen."
	for w in t.tokenize(etext, "en"):
		print w
	for w in t.tokenize(jtext, "ja"):
		print w
