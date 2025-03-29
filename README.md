

## 📌 项目结构

```
fedda_research_v1/
├── main.py                          # 主程序入口
├── config.py                        # 超参数配置
|
├── fedalgorithm/
|   |──  aggregation_factory.py      #聚合函数工厂
│   ├── fedadg.py                
│   ├── fedprox.py      
│   └── federated_averaging.py          
├── data/                            #放你想放的    
├── model/                           # 模型模块
│   ├── feature_extractor.py         # 特征提取器（ResNet50预训练）
│   ├── classifier.py                # 分类器
│   ├── domain_discriminator.py      # 域判别器
│   └── backbone_factory.py          # 多Backbone工厂
|
├── loss/                            # 损失函数模块
│   ├── mmd_loss.py                  # mmd损失（领域对齐损失）
│   ├── adversarial_loss.py          # 特征提取器和全局特征判别器之间的对抗损失
│   ├── consistency_loss.py          # 一致性正则化损失
│   └── distillation_loss.py         # 蒸馏损失
|
├── utils/                           # 工具模块
│   ├── cli_parser.py                # 命令行参数解析
│   ├── data_loader.py               # 本地数据加载器
│   └── logger.py                    # 日志工具
├── out/                             # 训练结果自动保存
├── requirements.txt                 # 依赖库清单
├── README.md                        # 项目说明文档
```

---

## 🚀 运行环境

推荐环境：
- Python 3.8+
- PyTorch >= 1.11
- torchvision
- pandas
- matplotlib
- tqdm

安装依赖：
```bash
pip install -r requirements.txt
```


**参数说明：**
| 参数 | 作用 |
|:-:|:-:|
| -d / --dataset | 数据集名称 (pacs, officehome) |
| --source | 源域子域名称 |
| --target | 目标域子域名称 |
| --clients | 客户端数量 |
| --epochs | 联邦通信轮数 |
| --local_epochs | 每个客户端本地训练轮数 |
| --device | 运行设备 (cuda:0 / cpu) |

---

## 🌐 实验流程框架

**整体流程图：**

```
          ┌──────────────┐
          │  数据输入与预处理 │
          └──────────────┘
                   ↓
        ┌──────────────────────────┐
        │  客户端本地训练 (Feature + Classifier + Discriminator) │
        └──────────────────────────┘
                   ↓
        ┌───────────────────────┐
        │  多损失联合训练 (Cls + MMD + Adv + Consistency + Distillation) │
        └───────────────────────┘
                   ↓
        ┌─────────────────────┐
        │  上传本地模型参数到服务器 │
        └─────────────────────┘
                   ↓
        ┌──────────────────────────┐
        │  服务器端聚合 (FedAvg / FedProx / FedAdg) │
        └──────────────────────────┘
                   ↓
        ┌────────────────────┐
        │  下发全局模型，进入下一轮 │
        └────────────────────┘
```

---

## 📄 输出结果

所有实验结果将自动保存至 **out/** 文件夹，包括：
- **log.txt**：训练过程日志
- **results.csv**：每轮各客户端准确率
- **curve.png**：准确率曲线图

---

## 🔥 已支持功能列表

FedAvg, FedProx, FedAdg 聚合策略  
MMD, Adversarial Loss, Consistency Loss, Distillation Loss  
支持 ResNet50, ViT 多Backbone  
支持 PACS, Office-Home 数据集  
Windows / Linux 双系统兼容  
实验日志 & 准确率曲线自动保存  
可扩展个性化联邦学习与隐私保护模块


