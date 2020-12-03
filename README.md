## 構成
basemodel_train.py: 学習用ファイル。学習後重みファイがmodels/に出力される。</br>
mnist_dataset.py: テストのデータセットを作成する。data/にsampleという名前でtxtファイルと、numpyファイルが出力される。</br>
save_bias_weight.py: 学習されたモデルを読み込み、c推論用に重みをtxt形式で保存。basemodelでは4層構造なので、4つの重みと4つのバイアスファイルが0~4の名前で出力される。</br>

cnn_2.cpp: sampleのtxtを読み込み、結果を出力する。</br>
methods.cpp: 畳み込み・ソフトマックス・relu・全結合が関数になっている。</br>


## 実行方法
`$ g++ cnn_2.cpp methods.cpp`

## 環境
tensorflow-gpu: 2.3.0