#! /usr/local/bin/python2.7
# -*- coding: utf-8 -*-
#
#【mt_api.py】
#
# 概要: encoder,decoder に基づく統計翻訳エンジンのWebAPI
#
import sys
import os
sys.path.append(".")
import urllib
import logging
import json
import mt_predict as P
from bottle import route, run, request
# logger の設定
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

# 大域変数(推定クラスの初期化などはコチラで実施)
# - 日英方向
dir_je = "./model_je"
model_je  = dir_je + "/je.model"
id2voc_je = dir_je + "/mt_id2voc.tsv"
voc2id_je = dir_je + "/mt_voc2id.tsv"

# - 英日方向
# dir_ej = "./model_ej"
# model_ej = model_dir + "/ej.model"
# id2voc_ej = model_dir + "/mt_id2voc.tsv"
# voc2id_ej = model_dir + "/mt_voc2id.tsv"
# decoder, encodfer インスタンス作成
# mt_ej = MtPredict(logging, False, model_ej, voc2id_ej, id2voc_ej)
mt_je = P.MtPredict(logging, False, model_je, voc2id_je, id2voc_je)

@route('/mt', method='POST')
def mt():
	res = {"validity": False}
	retval = []
	text = request.forms.get("text")
	lang = request.forms.get("lang")
	# URLデコード
	text = urllib.unquote(text)
	if lang == "je":
		text = mt_je.predict(text,"je")
	else:
	  return json.dumps(res)
	# 出力結果をjson にして返却
	res = {"validity": True, "text": text.replace(" ",'&nbsp;'), "lang": "je"}
	return json.dumps(res)

run(host='localhost', port=4610)
