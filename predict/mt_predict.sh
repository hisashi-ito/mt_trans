#! /bin/sh
#
# encoder, decoder 推定器
#
cmd="./mt_predict.py"
dir="model/model_je_10"
input=$1
model="${dir}/model"
voc2id="${dir}/mt_voc2id.tsv"
id2voc="${dir}/mt_id2voc.tsv"
main_cmd="${cmd} -i ${input} -m ${model} --voc2id ${voc2id} --id2voc ${id2voc}"
echo ${main_cmd}
eval ${main_cmd}
