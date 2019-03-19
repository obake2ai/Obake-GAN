# Obake-GAN (Perturbative GAN)
## Paper
https://arxiv.org/abs/1902.01514

## How To Use
1. Reedbush (or別のLinuxサーバー)にアクセス
2. pyenv上に以下パッケージ (requirements.txt) をpip install
3. otherGANs/data/直下にデータセット(mnist, cifar10, celeba, …)をDL
4. train_(データセット名).pyでcondition(バッチサイズ、モデルなど)を設定
5. (データセット名).run.shをqsub

## Hyper parameters
いじるハイパーパラメータは基本的に

・batchsize

・img_size（生成画像のサイズ）

・sample_interval（学習経過確認の頻度）

・modelsave_interval（モデル保存の頻度）

・log_interval（log記入の頻度list.txt）

・dataset（学習データセット名）

・saveDir（学習結果保存先、Noneであれば日付の入ったディレクトリを生成）

・resume（学習再開のインデックス、Noneならスクラッチ学習）

・loadDir（学習再開のモデル読込先）


・generator, discriminatorをmodels.py（Obake-GAN） or naiveresnet.py（PNN） or past_models.py（DCGAN, WGAN）から選択・読込

## datasets
公式のdataset_tool.pyの実装上、

otherGANs/data/(データセット名)/(サブディレクトリ)/image0001.jpg, …

という編成にする
