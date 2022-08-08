# lsm-intern-2022


# 準備
```bash
git clone git@github.com:oakeshott/lsm-intern-2022.git
```
## 要件

- Python 3.8.6

## Python Packageのインストール

```bash
pip install torch torchvision torchaudio
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
pip install -r requirements.txt
```

## サンプルプログラム

- `jupyter/mlp-example.ipynb`: DNNを利用した障害分類
- `jupyter/gcn-example.ipynb`: GNNを利用した障害分類

```
jupyter notebook
```
