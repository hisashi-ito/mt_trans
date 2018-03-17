## mt_trans  
encorder,decorder モデルを用いた翻訳モデルのchainer実装。  
実装内容は　Chainerによる実践深層学習 Ohmsha 新納浩幸 に記載のサンプルコードを学習、推定が実施しやすいように整理し、ミニバッチ学習を実施できるように改造した。　　

#### フォルダ構成  
1. train  
学習用コード  
1. predict  
推定(翻訳)用コード  
1. api  
bottle を用いたWebAPIコード　
1. BLEU  
精度評価用にBLEUを計算するコード
1. lib  
ライブラリ
