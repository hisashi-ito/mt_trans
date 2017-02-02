#! /bin/sh
cmd="./mt_train.py"
#input="./FREE_TRANSMAIL_1K_20161020.tsv"
input="./SMAll.tsv"
direction="f"
output="./model"
lang="je"
epoch_num=100
minibatch=3
step=10
# モデルディレクトリの作成
mkdir -p ${output}
main_cmd="${cmd} -i ${input} -d ${direction} -o ${output} -l ${lang} -e ${epoch_num} -s ${step} -m ${minibatch} -k 512"
echo ${main_cmd}
eval ${main_cmd}