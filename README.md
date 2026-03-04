配置环境：
安装miniconda（如已安装miniconda或anaconda则跳过此步骤）。安装教程：https://blog.csdn.net/weixin_39787913/article/details/145529639  

拉取代码 git clone https://github.com/ICIG/DGFNet.git 或者直接下载代码到本地，再上传到服务器。  

快速安装：conda create --name 新环境名 --file spec-file.txt，如果安装途中报错或安装完成后训练模型报错，则按照以下步骤逐步安装。
1. cd DGFNet
2. conda create --name DGF python=3.8
3. conda activate DGF
4. https://download.pytorch.org/whl/torch_stable.html 在这里下载 torch-1.10.0+cu102-cp38-cp38-linux_x86_64.whl; torchvision-0.11.1+cu102-cp38-cp38-linux_x86_64.whl; torchaudio-0.10.0+cu102-cp38-cp38-linux_x86_64.whl。下载完成后上传到服务器里以备后续安装。
5. pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
6. pip install torch-1.10.0+cu102-cp38-cp38-linux_x86_64.whl
7. pip install torchvision-0.11.1+cu102-cp38-cp38-linux_x86_64.whl
8. pip install torchaudio-0.10.0+cu102-cp38-cp38-linux_x86_64.whl
9. 使用如下命令安装detectron2  python -m pip install detectron2 -f \ https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html 安装文档页面如下https://detectron2-zhcn.readthedocs.io/zh-cn/latest/tutorials/install.html 
10. pip install -U opencv-python
11. 环境配置完成，后续如需要添加其他包，建议使用清华源下载，链接如下：https://pypi.tuna.tsinghua.edu.cn/simple  使用方法： pip install xxx -i https://pypi.tuna.tsinghua.edu.cn/simple

训练模型：  
在MUSIC数据集上训练
cd code
bash scripts/train_music.sh

在MUSIC21数据集上训练
cd code
bash scripts/train_music21.sh

训练结果在./data/ckpt中

数据集目录结构  
```text
data
├── audio
│   ├── acoustic_guitar
│   │   ├── M3dekVSwNjY.wav
│   │   └── ...
│   ├── trumpet
│   │   ├── STKXyBGSGyE.wav
│   │   └── ...
│   └── ...
│
├── frames
│   ├── acoustic_guitar
│   │   ├── M3dekVSwNjY.mp4
│   │   │   ├── 000001.jpg
│   │   │   └── ...
│   │   └── ...
│   ├── trumpet
│   │   ├── STKXyBGSGyE.mp4
│   │   │   ├── 000001.jpg
│   │   │   └── ...
│   │   └── ...
│   └── ...
│
├── detection_results
│   ├── acoustic_guitar
│   │   ├── M3dekVSwNjY.mp4.npy
│   │   └── ...
│   ├── trumpet
│   │   ├── STKXyBGSGyE.mp4.npy
│   │   └── ...
│   └── ...
│
└── motion_features
    ├── acoustic_guitar
    │   ├── M3dekVSwNjY.mp4.npy
    │   └── ...
    ├── trumpet
    │   ├── STKXyBGSGyE.mp4.npy
    │   └── ...
    └── ...
