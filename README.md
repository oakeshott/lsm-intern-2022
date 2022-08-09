# lsm-intern-2022

# 準備

```bash
git clone git@github.com:oakeshott/lsm-intern-2022.git
```

## 要件

- Python 3.8.6

## pyenv

```bash
# pyenv installation
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bash_profile
source ~/.bash_profile
# pyenv-virtualenv installation
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bash_profile
source ~/.bash_profile
```

```bash
pyenv install 3.8.6
pyenv virtualenv 3.8.6 AIML
pyenv local AIML
```

## Python Packageのインストール

```bash
pip install torch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1
pip install torch-scatter==2.0.7 torch-sparse==0.6.10 torch-cluster==1.5.9 torch-spline-conv==1.2.1 torch-geometric==2.0.1 -f https://data.pyg.org/whl/torch-1.9.1+cpu.html
pip install -r requirements.txt
```

## サンプルプログラム

- `jupyter/mlp-example.ipynb`: [DNNを利用した障害分類 (jupyter版)](jupyter/mlp-example.ipynb)
- `jupyter/gcn-example.ipynb`: [GNNを利用した障害分類 (jupyter版)](jupyter/gnn-example.ipynb)
- `mlp.py`: [DNNを利用した障害分類](mlp.py)
- `gnn.py`: [GNNを利用した障害分類](gnn.py)


## サンプルプログラムの実行

```bash
python mlp.py --train --test
python gnn.py --train --test
```
