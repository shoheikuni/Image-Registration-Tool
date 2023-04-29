# 概要

乳児の毎日の写真を繋げて成長記録動画を作りたかった。
全画像の乳児の位置を合わせるのは大変だ。
せめてベビーベッドの位置を合わせたい。
ベビーベッドは上から見ると長方形だ。

ということで、画像内の「長方形の枠」の位置合わせをするツールを作った。


# 動作環境
- Python 3


# 1. 各画像のアノテーション

特徴点抽出？
そんな難しいこと分かんない。
手で点を打てばいいだろう。

ツールを起動する。
```
python annotate.py
```
Fileメニューから、画像を格納したディレクトリを開く。  
全画像の枠の頂点に点を打つ。

操作方法：
- 矢印キー左またはA: 前の画像へ
- 矢印キー右またはD: 次の画像へ
- 矢印キー上またはW: 画像を左90度回転
- 矢印キー下またはS: 画像を右90度回転
- SPACE: 使用しない画像としてマーク
- マウス左クリック: 点を打つ
- マウス右クリック: 点を消す
- マウスホイールドラッグ: 画像表示位置の移動
- マウスホイールスクロール: 画像の拡大縮小
- SHIFT + マウスホイールスクロール: 表示明度調節
- CTRL + マウスホイールスクロール: 表示彩度調節

同ディレクトリに、各画像のアノテーション情報を格納したファイルが生成される。

# 2. 位置合わせ
枠のアスペクト比をあらかじめ物差しで測っておく。

ツールを実行する。
```
python fitter.py 入力ディレクトリ 枠のアスペクト比 出力ディレクトリ
```
出力ディレクトリに、位置合わせされた画像が出力される。日付入り。

入力画像において、枠が斜めに写っていてもよい。内部でホモグラフィ変換している。
デフォルトでは、枠を長方形に補正した画像が生成される。
入力画像の斜め具合が大きい場合、補正度合いが大きすぎて出力画像に違和感が出る。
そこで`--average-shape`オプションを使用すると、
全入力画像に渡る平均的な「斜め枠の写り具合」を計算し、それに合わせて補正をかける。

# 3. 画像を繋げる
ffmpegを使えば、連番の画像ファイルを繋げて動画ファイルを作成できる。

コマンド例:
```
ffmpeg -r 2 -i ./img_dir/%02d.jpg -vcodec libx264 -pix_fmt yuv420p -s 960x1280 out.mp4
```
`-r 2`は動画のfpsの指定。この場合2fps.   
`-i ./img_dir/%02d.jpg`は連番の入力画像ファイルで、この場合番号は2桁。  
`-vcodec libx264 -pix_fmt yuv420p`はコーデックの指定。  
`-s 960x1280` は動画の縦横サイズ指定。（入力画像サイズが揃っていれば未指定でいいかも）  
`out.mp4`は出力動画ファイル名。
