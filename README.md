# YOLOP-ONNX-Video-Inference-Sample
[YOLOP](https://github.com/hustvl/YOLOP)のPythonでのONNX推論サンプルです。<br>
ONNXモデルは、[hustvl/YOLOP/weights](https://github.com/hustvl/YOLOP/tree/main/weights) を使用しています。<br>

<img src="https://user-images.githubusercontent.com/37477845/149648575-208d2a82-ab6d-4f7e-8a6e-22b536a4413f.gif" width="45%">


# Requirement 
* OpenCV 3.4.2 or later
* onnxruntime 1.9.0 or later

# Demo
デモの実行方法は以下です。
```bash
python sample.py
```
* --video<br>
動画ファイルの指定<br>
デフォルト：video/sample.mp4
* --model<br>
ロードするモデルの格納パス<br>
デフォルト：weights/yolop-640-640.onnx
* --input_size<br>
モデルの入力サイズ<br>
デフォルト：640,640
* --score_th<br>
クラス判別の閾値<br>
デフォルト：0.3
* --nms_th<br>
NMSの閾値<br>
デフォルト：0.45

# Reference
* [hustvl/YOLOP](https://github.com/hustvl/YOLOP)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
YOLOP-ONNX-Video-Inference-Sample is under [MIT License](LICENSE).

# License(Movie)
サンプル動画は[NHKクリエイティブ・ライブラリー](https://www.nhk.or.jp/archives/creative/)の[中国・重慶　高速道路を走る](https://www2.nhk.or.jp/archives/creative/material/view.cgi?m=D0002050453_00000)を使用しています。
