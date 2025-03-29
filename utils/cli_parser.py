# 命令行参数解析器

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--epochs", type=int, default=20, help="Number of communication rounds")
    parser.add_argument("--clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--local_epochs", type=int, default=1, help="Local epochs per client per round")
    parser.add_argument("--source", type=str, required=True, help="Source domain name")
    parser.add_argument("--target", type=str, required=True, help="Target domain name")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., cuda:0 or cpu)")
    args = parser.parse_args()
    return args


