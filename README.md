# lsm-intern-2022


# 準備
```bash
git clone git@github.com:oakeshott/lsm-intern-2022.git
```
## 要件

- Python 3.8.6

## Python Packageのインストール

```bash
pip install torch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1
pip install torch-scatter==2.0.7 torch-sparse==0.6.10 torch-cluster==1.5.9 torch-spline-conv==1.2.1 torch-geometric==2.0.1 -f https://data.pyg.org/whl/torch-1.9.1+cpu.html
pip install -r requirements.txt
```

## サンプルプログラム

- `jupyter/mlp-example.ipynb`: DNNを利用した障害分類
- `jupyter/gcn-example.ipynb`: GNNを利用した障害分類

```
jupyter notebook
```
