# 命令行参数解析器

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True, help="数据集")
    parser.add_argument("--epochs", type=int, default=20, help="大epoch循环轮次")
    parser.add_argument("--clients", type=int, default=3, help="客户端数量")
    parser.add_argument("--local_epochs", type=int, default=1, help="每个客户端本地epoch数量")
    parser.add_argument("--source", type=str, required=True, help="源域名称")
    parser.add_argument("--target", type=str, required=True, help="目标域名称")
    parser.add_argument("--device", type=str, default="cuda:0", help="使用的设备 (e.g., cuda:0 or cpu)")
    parser.add_argument('--noniid', action='store_true', help='是否使用Non-IID模式')
    parser.add_argument('--noniid_ratio', type=float, default=0.5, help='Non-IID分布比例 (0-1)')

    args = parser.parse_args()
    return args


