# DL_CTR

<<<<<<< HEAD
### 注：data文件夹用来存放文件，后续的DeepFM、FM等模型需要使用GBDT_LR文件夹内feature_engineer文件夹处理的特征矩阵。
1、GBDT_LR模型：将连续特征通过GBDT离散化并结合原类别特征，使用CSR矩阵存储One-Hot以后的特征矩阵，进而用LR模型训练。
2、DeepFM模型参考https://github.com/ChenglongChen/tensorflow-DeepFM，其中FM模型的实现只需要将DeepFM中的Deep部分去掉即可实现。
安利一个推荐系统的博客，https://www.jianshu.com/p/6f1c2643d31b
=======
### 一、data文件夹用来存放文件，后续的DeepFM、FM等模型需要使用GBDT_LR文件夹内feature_engineer文件夹处理的特征矩阵。
DeepFM模型参考https://github.com/ChenglongChen/tensorflow-DeepFM
FM模型的实现只需要将DeepFM中的Deep部分去掉即可实现，
>>>>>>> 2265e5522f46bad58cf590460a53c3dab1284f47
