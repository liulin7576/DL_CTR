# DL_CTR

### 注：data文件夹用来存放特征矩阵，后续的DeepFM、FM等模型需要使用GBDT_LR文件夹内feature_engineer文件夹处理的特征矩阵。
  
  1、数据集下载：[科大讯飞初赛数据集下载](http://www.dcjingsai.com/common/cmpt/2018%E7%A7%91%E5%A4%A7%E8%AE%AF%E9%A3%9EAI%E8%90%A5%E9%94%80%E7%AE%97%E6%B3%95%E5%A4%A7%E8%B5%9B_%E7%AB%9E%E8%B5%9B%E4%BF%A1%E6%81%AF.html)


  2、GBDT_LR模型：将连续特征通过GBDT离散化并结合原类别特征，使用CSR矩阵存储One-Hot以后的特征矩阵，进而用LR模型训练。
  
  3、DeepFM模型参考[DeepFM参考链接](https://github.com/ChenglongChen/tensorflow-DeepFM)，其中FM模型的实现只需要将DeepFM中的Deep部分去掉即可实现。安利一个推荐系统的博客，[推荐系统博客](https://www.jianshu.com/p/6f1c2643d31b)。
  
  4、Deep & Cross Network(DCN)模型参考DeepFM，实现EarlyStopping、L2正则化、Dropout，详见DCN文件夹。
  
  5、Product-based Neural Networks(PNN)模型参考DeepFM，实现EarlyStopping、Dropout、inner-product和outer-product，通过use_inner参数来调用，详见PNN文件夹。
  
  6、Neural Factorization Machines(NFM)模型参考DeepFM，实现EarlyStopping、Dropout，详见NFM文件夹。

  7、Attentional Factorization Machines(AFM)模型参考NFM，详见AFM文件夹。
  
  
### 注：NFM、DeepFM训练模型速度较快；
