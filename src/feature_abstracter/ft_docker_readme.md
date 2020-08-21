
### 功能
Docker FeatureComparator 通过封装Rest API服务，提供视频查询服务。
Docker FeatureAbstracter 通过封装Rest API服务，提供提取视频特征服务。
Docker FrameAbstracter   通过封装Rest API服务，提供提取视频抽帧服务。
其它辅助容器：MySQL等。

#### 特征比对 Docker FeatureComparator
主要接口：
POST /api/v1/video/feature/search
输入：待检测文件特征值。
输出：片名、所在帧位置、相似度。

#### 特征提取 Docker FeatureAbstracter
POST /api/v1/video/feature/abstract
输入：视频文件路径
输出：特征值 / 不存路径 / 直接写入数据库 ？ 【建议返回特征值】

POST /api/v1/video/feature/batch/abstract
输入：视频文件目录的路径
输出：视频总数、
     成功提取数量、成功提取的文件、文件特征值、
     失败提取数量、失败提取的文件
     
