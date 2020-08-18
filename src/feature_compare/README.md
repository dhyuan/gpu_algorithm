# gpu_algorithm



### features_loader.py
负责加载features信息。


### features_comparator.py
利用GPU在source features中比对 test features，找到最相似的位置。





# 准备环境

    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
    conda config --set show_channel_urls yes

    # 创建基于环境 pytho3.7 的 tensorflow 环境
    conda create -n tf1.15_py3.7 python=3.7
    
    # 
    conda activate tf1.15_py3.7
    conda install --channel https://conda.anaconda.org/anaconda tensorflow=1.15.0
    
    # 重命名tensorflow1.5 为tf1.15_py3.7
    conda create -n tf1.15_py3.7 --clone tensorflow1.5
    conda active tf1.15_py3.7
    conda remove -n tensorflow1.5 --all


