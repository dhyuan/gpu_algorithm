

https://docs.docker.com/engine/install/centos/

### 在CentOS 7 安装Docker

#### 添加docker repository
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo http://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo


#### 安装 docker
yum remove docker \
                  docker-client \
                  docker-client-latest \
                  docker-common \
                  docker-latest \
                  docker-latest-logrotate \
                  docker-logrotate \
                  docker-engine
yum install -y docker-ce-19.03.12 docker-ce-cli-19.03.12 containerd.io


#### 
mkdir -p /home/docker/var/lib/docker
mv /var/lib/docker /home/docker/var/lib/docker
ln -s /home/docker/var/lib/docker /var/lib/docker

#### 启动 docker
systemctl enable docker

vi /usr/lib/systemd/system/docker.service
更改image存放目录(因为当前/分区太小) 在ExecStart=/usr/bin/dockerd 后面添加： --graph /home/docker/var/lib/docker 

systemctl daemon-reload 
systemctl start docker
systemctl status docker



#### 下载镜像  
docker pull tensorflow/tensorflow:1.15.0-gpu-py3
docker pull nvcr.io/nvidia/tensorflow:20.07-tf1-py3

https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel_19.12.html#rel_19.12

#### 安装  nvidia-container-toolkit, 
> On versions including and after 19.03, you will use the nvidia-container-toolkit package and the --gpus all flag.
> https://github.com/NVIDIA/nvidia-docker
>

    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo
#### 测试 TensorFlow docker    
docker run --gpus all -it --rm tensorflow/tensorflow:1.15.0-gpu-py3 \
   python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

#### 构建应用镜像
docker build -t v_ai/feature_abstracter-gpu:0.1 .

#### 打包、导入镜像
docker save -o feature_abstracter_0.1.tar.gz v_ai/feature_abstracter:0.1
docker load < feature_abstracter_0.1.tar.gz


#### 运行 
> https://hub.docker.com/r/tensorflow/tensorflow/
>

##### Run on CPU
docker run -it --rm tensorflow/tensorflow:1.15.2-gpu-py3 bash

##### Run on GPU
docker run -it --rm --runtime=nvidia tensorflow/tensorflow:1.15.2-gpu-py3 python

docker run -it --rm -v ./tmp:/tmp tensorflow/tensorflow:1.15.2-gpu-py3 bash


