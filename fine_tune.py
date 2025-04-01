import torch
import torch.nn as nn
import torch.optim as optim
from model.feature_extractor import FeatureExtractor
from model.classifier import Classifier
from utils.data_loader import get_loader
from utils.cli_parser import get_args
import config
from tqdm import tqdm
import os
#微调阶段只针对目标域分类器，冻结特征提取器，避免过拟合

# 评估函数
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


def fine_tune():
    args = get_args()
    dataset = args.dataset.lower()
    target_domain = args.target
    device = torch.device(args.device)

    print("\n🎯 Fine-tune 微调阶段开始...")

    # === 加载训练好的全局模型参数 ===
    extractor = FeatureExtractor().to(device)
    classifier = Classifier().to(device)
    extractor.load_state_dict(torch.load("out/global_extractor.pth"))
    classifier.load_state_dict(torch.load("out/global_classifier.pth"))

    # === 冻结特征提取器参数，只训练分类器 ===
    for param in extractor.parameters():
        param.requires_grad = False

    # === 优化器仅更新分类器 ===
    optimizer = optim.Adam(classifier.parameters(), lr=config.FINETUNE_LR)
    criterion = nn.CrossEntropyLoss()

    # === 准备目标域数据 ===
    target_loader = get_loader(os.path.join(os.getcwd(), "data", dataset), target_domain, batch_size=config.BATCH_SIZE)

    # === Fine-tune 开始 ===
    classifier.train()
    for epoch in range(args.epochs):
        total_loss, total_acc = 0, 0
        data_iterator = tqdm(target_loader, desc=f"Fine-tune Epoch {epoch + 1}")
        for x, y in data_iterator:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                feat = extractor(x)
            logits = classifier(feat)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += (logits.argmax(1) == y).float().mean().item()
            data_iterator.set_postfix(loss=total_loss / (len(target_loader)),
                                      acc=total_acc / (len(target_loader)))

        print(f"Epoch {epoch + 1}: Loss={total_loss / len(target_loader):.4f}, "
              f"Acc={total_acc / len(target_loader) * 100:.2f}%")

    # === 保存微调后的分类器参数 ===
    torch.save(classifier.state_dict(), "out/fine_tuned_classifier.pth")
    print("✅ Fine-tune 完成，分类器已保存 → out/fine_tuned_classifier.pth")

    # === 🎯 微调后验证阶段 ===
    print("\n🎯 开始目标域验证评估...")
    extractor.eval()
    classifier.eval()
    final_acc = evaluate(extractor, classifier, target_loader, device)
    print(f"✅ 目标域验证准确率：{final_acc * 100:.2f}%")

if __name__ == "__main__":
    fine_tune()


