

## 📌 项目结构

```
project/
├── main.py                  → 联邦训练主程序
├── fine_tune.py             → 微调程序
├── config.py                → 超参数配置
├── model/                   → 模型模块
│   ├── backbone_factory.py
│   ├── classifier.py
│   ├── domain_discriminator.py
│   └── feature_extractor.py
├── loss/                    → 损失函数模块
│   ├── adversarial_loss.py
│   ├── consistency_loss.py
│   ├── distillation_loss.py
│   └── mmd_loss.py
├── utils/                   → 工具模块
│   ├── cli_parser.py
│   ├── data_loader.py
│   └── logger.py
└── fedalgorithm/            → 聚合策略模块
    ├── aggregation_factory.py
    ├── fedadg.py
    ├── federated_averaging.py
    └── fedprox.py

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



