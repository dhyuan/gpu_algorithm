

https://docs.docker.com/engine/install/centos/


### 在CentOS 7 安装Docker

#### 添加docker repository
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo http://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo


#### 下载镜像
sudo yum install docker-ce-19.03.12 docker-ce-cli-19.03.12 containerd.io
sudo yum install docker-ce-19.03.12 docker-ce-cli-19.03.12 containerd.io

#### 启动 docker
systemctl start docker
systemctl enable docker
systemctl status docker

docker pull tensorflow/tensorflow:1.15.2-gpu-py3

