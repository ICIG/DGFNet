配置环境：
安装miniconda（如已安装miniconda或anaconda则跳过此步骤）
1. 拉取代码
2. cd DGFNet
3. conda create --name DGF python=3.8
4. conda activate DGF
5. https://download.pytorch.org/whl/torch_stable.html 在这里下载 torch-1.10.0+cu102-cp38-cp38-linux_x86_64.whl; torchvision-0.11.1+cu102-cp38-cp38-linux_x86_64.whl; torchaudio-0.10.0+cu102-cp38-cp38-linux_x86_64.whl。下载完成后上传到服务器里以备后续安装。
6. pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
7. pip install torch-1.10.0+cu102-cp38-cp38-linux_x86_64.whl
8. pip install torchvision-0.11.1+cu102-cp38-cp38-linux_x86_64.whl
9. pip install torchaudio-0.10.0+cu102-cp38-cp38-linux_x86_64.whl
10. 使用如下命令安装detectron2  python -m pip install detectron2 -f \ https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html 安装文档页面如下https://detectron2-zhcn.readthedocs.io/zh-cn/latest/tutorials/install.html 
11. pip install -U opencv-python
12. 环境配置完成
