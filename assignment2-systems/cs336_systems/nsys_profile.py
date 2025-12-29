import argparse
import time
import torch
from tqdm import tqdm

from cs336_basics.ch3.transformer_lm import TransformerLM
from cs336_basics.ch4.cross_entropy import cross_entropy
from cs336_basics.ch4.adamw import AdamW


def main(args, device, uid):
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)

    model_args = dict(vocab_size=args.vocab_size, d_model=args.d_model, context_length=args.context_length, num_layers=args.num_layers, num_heads=args.num_heads, theta=args.theta, d_ff=args.d_ff, device=device)
    model = TransformerLM(**model_args).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), eps=args.eps)

    print(f"Warm up {args.warm_up} steps on {device}...")
    for _ in tqdm(range(1, 1+args.warm_up)): 
        optimizer.zero_grad(set_to_none=True) # 性能优化：set_to_none 比 zero_ 更快

        logits = model(x)
        loss = cross_entropy(logits, x)
        loss.backward()

        optimizer.step()

    
    print(f"Benchmarking {args.max_iters} steps on {device}...")
    for _ in tqdm(range(1, 1+args.max_iters)):
        with torch.cuda.nvtx.range('forward'):
            logits = model(x)
            loss = cross_entropy(logits, x)
        with torch.cuda.nvtx.range('backward'):
            optimizer.zero_grad(set_to_none=True) # 性能优化：set_to_none 比 zero_ 更快
            loss.backward()
        with torch.cuda.nvtx.range('optimize'):
            optimizer.step()
        torch.cuda.synchronize()
    

# --- 3. 命令行参数解析 ---
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark")
    # 模型超参 (与你的 GPT 类定义对应)
    parser.add_argument("--context_length", type=int, default=256)  # 要变
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--theta", type=float, default=10000.0)   # 姑且固定
    parser.add_argument("--vocab_size", type=int, default=10000)  # 这里固定
    # 训练超参
    parser.add_argument("--both", type=bool, default=False)
    parser.add_argument("--warm-up", type=int, default=5)
    parser.add_argument("--max_iters", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)  # 这里固定
    ## Optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    uid = time.strftime('%m%d%H%M')
    device = 'cuda' if torch.cuda.is_available() else 'mps'
    main(args, device, uid)