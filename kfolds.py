from pathlib import Path
from sklearn.model_selection import KFold, train_test_split


datapath = 'gsk-ene-1.1-bccwj-json-jumanpp-type/bccwj-ene-jumanpp-type-ene-single.txt'  # 'bccwj-ene-jumanpp-type.txt'
k = 5
with open(datapath) as f:
    data = f.read().split('\n\n')
p = Path(f'NERdata_ene_single_cv_{k}')
p.mkdir(exist_ok=True)
# 各クラス毎に train:test比を保ちつつ分割
# n_splits はクラスサンプル数の最小以下

kf = KFold(n_splits=k, shuffle=True, random_state=0)
train_test_splits = list(kf.split(data))
for k, (train_dev_idx, test_idx) in enumerate(train_test_splits):
    train_idx, dev_idx = train_test_split(train_dev_idx,
                                          test_size=len(test_idx),
                                          random_state=0)

    op = p / Path(f'NERdata_{k}')
    op.mkdir(exist_ok=True)
    with open(op / Path('train.txt'), 'w') as f:
        for i in train_idx:
            f.write(data[i])
            f.write('\n\n')
    with open(op / Path('dev.txt'), 'w') as f:
        for i in dev_idx:
            f.write(data[i])
            f.write('\n\n')
    with open(op / Path('test.txt'), 'w') as f:
        for i in test_idx:
            f.write(data[i])
            f.write('\n\n')
