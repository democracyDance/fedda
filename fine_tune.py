import torch
import torch.nn as nn
import torch.optim as optim
from model.feature_extractor import FeatureExtractor
from model.classifier import Classifier
from utils.data_loader import get_loader
from utils.cli_parser import get_args
import config
import os

# ========== 验证函数 ==========
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

# ========== 微调主函数 ==========
def fine_tune(dataset, target_domain, epochs, device):
    print(f"Fine-tune阶段启动 → Target Domain: {target_domain}")

    data_dir = os.path.join(os.getcwd(), "data", dataset)
    target_loader = get_loader(data_dir, target_domain, batch_size=config.BATCH_SIZE)

    # 加载联邦训练后保存的全局模型
    extractor = FeatureExtractor().to(device)
    classifier = Classifier().to(device)
    extractor.load_state_dict(torch.load('out/global_extractor.pth'))
    classifier.load_state_dict(torch.load('out/global_classifier.pth'))

    # 定义优化器 & 损失函数
    optimizer = optim.Adam(list(extractor.parameters()) + list(classifier.parameters()), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 微调训练
    for epoch in range(epochs):
        extractor.train()
        classifier.train()
        total_loss, total_acc = 0, 0
        for x, y in target_loader:
            x, y = x.to(device), y.to(device)
            feat = extractor(x)
            logits = classifier(feat)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc += (logits.argmax(1) == y).float().mean().item()
        avg_loss = total_loss / len(target_loader)
        avg_acc = total_acc / len(target_loader)
        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Acc={avg_acc * 100:.2f}%")

    # 目标域最终验证
    final_acc = evaluate(extractor, classifier, target_loader, device)
    print(f" Fine-tune完成 → 目标域验证准确率：{final_acc * 100:.2f}%")

    # 保存微调后模型
    torch.save(extractor.state_dict(), "out/final_extractor.pth")
    torch.save(classifier.state_dict(), "out/final_classifier.pth")
    print("微调模型已保存：out/final_extractor.pth, out/final_classifier.pth")


if __name__ == "__main__":
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fine_tune(dataset=args.dataset, target_domain=args.target, epochs=5, device=device)
