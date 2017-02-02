#! /usr/local/bin/ruby19
# -*- coding: utf-8 -*-
#
# Unpublished Copyright(C) NTT Resonat 2016
#
#【BLEU】
#
# 概要: 統計機械翻訳の精度指標であるBLEU値を計算する
#       本計算では出力と入力文は事前に<SP>にて
#       tokenizeされていることを想定とする
#
#       BLEUの標識は以下のURLを参照
#       http://unicorn.ike.tottori-u.ac.jp/2010/s072046/paper/graduation-thesis/node32.html
#       http://phontron.com/paper/neubig13nl212.pdf
#
# usage: ./BLEU.rb -i <input>
#                  -o <output>
#                  -n <ngram> n-gramまで考慮するか
#                             デフォルトは 4
#
require 'logger'
require 'optparse'

module CONST
  BOS = "<BOS>"
  EOS = "<EOS>"
  COMBINE_STR = "＋"
end

class BLEU
  include CONST
  def initialize(logger, input, num_gram)
    @logger = logger
    @input = input
    @num_gram = numgram
    # BLEU の計算で利用する数値
    @c = 0 # テストデータ全体における翻訳文のtoken 長
    @r = 0 # テストデータ全体における参照文のtoken 長
    @ngram_overlap = {}
  end
  
  def inc_tokens(text, mode)
    if mode == "c"
      @c += text.split("\s",-1).size
    else
      @r += text.split("\s",-1).size
    end
  end

  def overlap(rel, test)
    _test = tesgt.dup
    # test のngramの数
    t = test.size()
    # n-gramの共通数
    cnt = 0
    rel.each do |ngram|
      idx =  _test.index(ngram)
      if idx
        cnt += 1
        _test.delete_at(idx)
      end
    end
    return [cnt, t]
  end
  
  # 指定したnにともなってngram配列作成する
  def to_ngram(text, n)
    ary = []
    tokens = text.split("\s",-1)
    # 終端記号付与
    tokens.unshift(BOS)
    tokens.push(EOS)
    return [] if tokens.size < n
    k = tokens.size - n + 1
    k.times do |idx|
      ary << tokens[idx..(idx + n - 1)].join(COMBINE_STR).dup
    end
    return ary
  end
  
  def bp()
    # bleu値を計算
    # - ペナルティ項を計算
    x = 0.0
    if @c >= @r
      x = 1.0
    else
      x = Math.exp(1.0 - (@r/@c))
    end
    return x
  end
  
  def perform()
    f = open(@input, "r")
    while line = f.gets
      # 入力文, 正解文、翻訳文
      i_text , r_text, t_text  = line.chomp.force_encoding("utf-8").split("\t",3)
      next if i_text.to_s.emtpy? || c_text.to_s.empty? || t_text.to_s.empty?
      
      # tokenの数を計算
      # 翻訳文で@c をインクリメント
      inc_tokens(t_text, "c")
      # 正解文で@tau をインクリメント
      inc_tokens(r_text, "r")
      
      # n-ngram の加重平均を計算する
      1.upto(@num_gram).each do |n|
        r_ngram = to_ngram(r_text,n)
        t_ngram = to_ngram(t_text,n)
        cnt, t = overlap(r_ngram, t_ngram)
        unless @ngram_overlap.has_key?(n)
          @ngram_overlap[n] = [cnt, t]
        else
          @ngram_overlap[n][0] += cnt
          @ngram_overlap[n][0] += t
        end
      end # ngram
    end # liine
    f.close
    
    # BLUE計算
    score = 0.0
    @ngram_overlap.sort.each do |n, ary|
      score += (Math.log(ary[0].to_f/ary[1].to_f) / n.to_f)
    end
    score = bp() * Math.exp(score)
    $stdout.puts "BLRU: #{score}"
  end # perform
end # BLEU

def main(argv)
  logger = Logger.new(STDERR)
  logger.level = Logger::INFO
  param = ARGV.getopts("i:n:")
  input  = param["i"]
  num_gram = param["n"]
  if input.nil? || num_gram.nil?
    $stderr.puts "usage: BLEU.rb -i <input> -n <ngram>"
    exit 1
  end
  logger.info("***start BLEU***")
  BLEU.new(logger, input, num_gram).perform()
  logger.info("***start BLEU***")
end

if __FILE__ == $0
  main(ARGV)
end
