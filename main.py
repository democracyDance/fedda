import torch
import torch.nn as nn
import torch.optim as optim
from model.feature_extractor import FeatureExtractor
from model.classifier import Classifier
from model.domain_discriminator import DomainDiscriminator
from fedalgorithm.federated_averaging import federated_average  # 聚合函数用fedavg
from utils.data_loader import get_loader
from utils.logger import Logger
from utils.cli_parser import get_args
from loss.mmd_loss import compute_mmd
from loss.adversarial_loss import adversarial_loss
from loss.consistency_loss import consistency_loss
from loss.distillation_loss import distillation_loss
import config
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import subprocess
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from utils.data_loader import split_noniid
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
# 源域验证函数
def evaluate(model, classifier, dataloader, device):
    model.eval()
    classifier.eval()
    total_acc = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            feat = model(x)
            logits = classifier(feat)
            acc = (logits.argmax(1) == y).float().mean().item()
            total_acc += acc
    return total_acc / len(dataloader)

# 每个客户端训练时，输入的参数是客户端id、数据集、源域和目标域、本地训练批次、设备、日志打印与保存,每个客户的数据
def client_update(client_id, global_model, dataset, source_domain, target_domain, local_epochs, device, logger, client_dataset):
    # 每个客户端都有自己的模型副本
    feature_extractor = FeatureExtractor().to(device)
    classifier = Classifier().to(device)
    discriminator = DomainDiscriminator().to(device)

    # 确保每轮通信开始前，客户端参数与服务器一致
    feature_extractor.load_state_dict(global_model['extractor'])
    classifier.load_state_dict(global_model['classifier'])

    # 本地训练优化器，用adam同时优化 FeatureExtractor + Classifier + Discriminator
    optimizer = optim.Adam(
        list(feature_extractor.parameters()) +
        list(classifier.parameters()) +
        list(discriminator.parameters()),
        lr=config.LEARNING_RATE
    )

    # 动态根据命令行参数选择数据集 & 源域source/目标域target
    data_dir = os.path.join(os.getcwd(), "data", dataset)
    #source_loader = get_loader(data_dir, source_domain, batch_size=config.BATCH_SIZE)
    source_loader = DataLoader(client_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    target_loader = get_loader(data_dir, target_domain, batch_size=config.BATCH_SIZE)

    # 交叉熵分类损失，同时将所有模块定义为训练模式
    criterion = nn.CrossEntropyLoss()
    feature_extractor.train()
    classifier.train()
    discriminator.train()

    # 开始本地训练
    for epoch in range(local_epochs):
        total_loss, total_acc = 0, 0
        # tqdm进度条
        data_iterator = tqdm(zip(source_loader, target_loader),
                             desc=f"Client {client_id} Epoch {epoch + 1}",
                             leave=True)
        for (src_x, src_y), (tgt_x, _) in data_iterator:
            # 将数据移至GPU
            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)

            # 前向传播,特征提取与分类
            src_feat = feature_extractor(src_x)
            tgt_feat = feature_extractor(tgt_x)
            src_logits = classifier(src_feat)
            tgt_logits = classifier(tgt_feat)

            # 分类损失 + MMD损失
            cls_loss = criterion(src_logits, src_y)
            mmd = compute_mmd(src_feat, tgt_feat)

            # 对抗损失
            adv_preds = discriminator(torch.cat([src_feat, tgt_feat]))
            adv_labels = torch.cat([torch.ones(src_feat.size(0), 1), torch.zeros(tgt_feat.size(0), 1)]).to(device)
            adv_loss = adversarial_loss(adv_preds, adv_labels)

            # 一致性正则，拿原始数据进行扰动后，扰动前和扰动后数据都扔进特征提取器和分类器，然后看分类结果一样不
            tgt_x_aug = torch.flip(tgt_x, dims=[-1])
            tgt_feat_aug = feature_extractor(tgt_x_aug)
            tgt_logits_aug = classifier(tgt_feat_aug)
            cons_loss = consistency_loss(tgt_logits, tgt_logits_aug)

            # 知识蒸馏
            #teacher_logits 是冻结版，提供“过去的预测”，tgt_logits 是当前模型最新预测
            #distill_loss 计算 tgt_logits 与 teacher_logits 的均方误差
            #反向传播和优化时，更新的是当前模型的参数 → 让它接近 teacher_logits
            #Self-Distillation 引入了一个额外的 平滑目标：当前预测 ≈ 过去预测
            #不依赖标签 → 只约束「当前预测 ≈ 过去预测」，一言以蔽之，是个正则结构
            with torch.no_grad():
                teacher_logits = tgt_logits.detach()
            distill = distillation_loss(tgt_logits, teacher_logits)

            # 总损失
            total = (cls_loss + config.LAMBDA_MMD * mmd + config.LAMBDA_ADV * adv_loss +
                     config.LAMBDA_CONSISTENCY * cons_loss + config.LAMBDA_DISTILLATION * distill)

            # 反向传播
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            # 累计损失与准确率
            total_loss += total.item()
            total_acc += (src_logits.argmax(1) == src_y).float().mean().item()
            data_iterator.set_postfix(loss=total_loss / (len(source_loader)),
                                      acc=total_acc / (len(source_loader)))

        avg_loss = total_loss / len(source_loader)
        avg_acc = total_acc / len(source_loader)
        logger.log(f"[Client {client_id}] Epoch {epoch + 1} - Loss: {avg_loss:.4f} - Acc: {avg_acc * 100:.2f}%")

    return feature_extractor.state_dict(), classifier.state_dict(), avg_acc

# 准确率曲线保存
def plot_results(results, out_dir):
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    plt.figure()
    for client_id in range(len(df.columns) - 1):
        plt.plot(df["Round"], df[f"Client{client_id}_Acc"], label=f"Client{client_id}")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "curve.png"))
    plt.close()

# =====================  主函数  =====================
def main():
    args = get_args()
    dataset = args.dataset.lower()
    source_domain = args.source
    target_domain = args.target
    epochs = args.epochs
    num_clients = args.clients
    local_epochs = args.local_epochs
    device = torch.device(args.device)
    #日志与结果文件
    os.makedirs("out", exist_ok=True)
    logger = Logger("out")
    logger.log(f"Dataset: {dataset}, Source: {source_domain}, Target: {target_domain}, "
               f"Epochs: {epochs}, Clients: {num_clients}, Local Epochs: {local_epochs}, Device: {device}")

    #接下来是划分iid和non-iid

    # 源域数据路径 & transform
    #它的作用是 → 自动帮你拼好你的数据集所在文件夹的绝对路径D:/fed_code/fedda_research_v1/data/pacs
    data_dir = os.path.join(os.getcwd(), "data", dataset)
    #在你用 ImageFolder 加载数据时，每张图片都会自动经过这个 transform → 变成训练用的张量数据
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 加载完整源域数据
    full_dataset = ImageFolder(root=f"{data_dir}/{source_domain}", transform=transform)

    if args.noniid:
        indices = split_noniid(full_dataset, num_clients, ratio=args.noniid_ratio)
    else:
        indices = np.array_split(range(len(full_dataset)), num_clients)

    # 划分到各客户端
    client_datasets = [Subset(full_dataset, idxs) for idxs in indices]

    # 初始化全局模型
    #每次运行都会 new 一个新的 FeatureExtractor() 和 Classifier() → 参数被初始化→ 不会自动加载之前训练好的参数
    global_model = {
        'extractor': FeatureExtractor().state_dict(),
        'classifier': Classifier().state_dict()
    }

    # 准确率记录字典
    results = {"Round": []}
    for client_id in range(num_clients):
        results[f"Client{client_id}_Acc"] = []

    # ========== 联邦训练开始 ==========
    for round in range(epochs):
        client_models = []
        accs = []
        logger.log(f"\n=== Round {round + 1} ===")

        # 把 client_datasets 传给 client_update
        for client_id in range(num_clients):
            extractor_state, classifier_state, acc = client_update(
                client_id, global_model, dataset, source_domain, target_domain, local_epochs, device, logger, client_datasets[client_id])
            model = FeatureExtractor()
            model.load_state_dict(extractor_state)
            client_models.append(model)
            accs.append(acc)

        # FedAvg 聚合
        avg_model = federated_average(client_models)
        global_model['extractor'] = avg_model.state_dict()
        logger.log(f"Round {round + 1} aggregation done.")

        results["Round"].append(round + 1)
        for i, acc in enumerate(accs):
            results[f"Client{i}_Acc"].append(acc)

    # 保存曲线
    plot_results(results, "out")
    logger.log("Training finished. Results saved in out/")

    # ========== 新增：训练结束后，源域验证 ==========
    logger.log("开始源域验证评估...")
    source_loader = get_loader(os.path.join(os.getcwd(), "data", dataset), source_domain, batch_size=config.BATCH_SIZE)
    extractor = FeatureExtractor().to(device)
    classifier = Classifier().to(device)
    extractor.load_state_dict(global_model['extractor'])
    classifier.load_state_dict(global_model['classifier'])
    #目前只在源域进行验证
    source_acc = evaluate(extractor, classifier, source_loader, device)
    logger.log(f"源域最终验证准确率：{source_acc * 100:.2f}%")

    # ========== 保存全局模型 ==========
    torch.save(global_model['extractor'], "out/global_extractor.pth")
    torch.save(global_model['classifier'], "out/global_classifier.pth")

    # ========== 新增：自动调用 Fine-tune ==========
    logger.log("联邦训练结束，自动启动 Fine-tune 微调阶段...")
    subprocess.call(["python", "fine_tune.py",
                     "--dataset", dataset,
                     "--source", source_domain,#无所谓，只是为了满足utils/cli_parser.py的输入格式
                     "--target", target_domain,
                     "--epochs", "20",
                     "--device", args.device])


if __name__ == "__main__":
    main()


