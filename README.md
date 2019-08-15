# bert-ner

## データセット
- `input/` を参考
- `input/labels_enesub.txt` は教師データに合わせて準備する

## 依存リソース
- BERT事前学習モデル: [BERT日本語Pretrainedモデル(京大)](http://nlp.ist.i.kyoto-u.ac.jp/index.php?BERT%E6%97%A5%E6%9C%AC%E8%AA%9EPretrained%E3%83%A2%E3%83%87%E3%83%AB)
  - `Japanese_L-12_H-768_A-12_E-30_BPE`
  - NERなので単語単位のBERTを採用している

- 知識ベースとして用いる [「拡張固有表現＋Wikipedia」データ](http://www.languagecraft.com/enew/)
  - `ENEW_ENEtag_20160305.txt`
  - `ENEW_StructuredWikipedia_20160305.txt`

## 導入
- install `pipenv`
- `pipenv install`