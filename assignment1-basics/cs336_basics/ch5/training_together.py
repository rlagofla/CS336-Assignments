import argparse
import os
import time
import numpy as np
import torch
import wandb
from tqdm import tqdm

from .data_loading import get_batch
from ..ch3.transformer_lm import TransformerLM
from ..ch4.cross_entropy import cross_entropy
from ..ch4.adamw import AdamW
from .checkpointing import save_checkpoint, load_checkpoint
from ..ch4.learning_rate_schedule import lr_cosine_schedule
from ..ch4.gradient_clipping import gradient_clipping


def main(args, device, run, uid):
    os.makedirs(args.out_dir, exist_ok=True)
    
    # A. 加载数据 (uint16 memmap)
    train_data = np.memmap(os.path.join(args.data_dir, f"{args.dataset}_train.bin"), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(args.data_dir, f"{args.dataset}_valid.bin"), dtype=np.uint16, mode='r')

    # B. 初始化模型 (假设你的模型类叫 MyGPT)
    model_args = dict(vocab_size=args.vocab_size, d_model=args.d_model, context_length=args.context_length, num_layers=args.num_layers, num_heads=args.num_heads, theta=args.theta, d_ff=args.d_ff, device=device)
    model = TransformerLM(**model_args).to(device)

    # C. 初始化优化器 (使用你写的 AdamW)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2), eps=args.eps)

    # D. 断点续传逻辑
    iter_num = 0
    if args.resume:
        print(f"Resuming from {args.resume}...")
        resume_path = os.path.join(args.out_dir, args.resume)
        iter_num = load_checkpoint(resume_path, model, optimizer)

    # E. 训练循环
    print(f"Starting training on {device}...")
    
    pbar = tqdm(range(1+iter_num, 1+args.max_iters))
    for step in pbar:
        
        # 1. 联动学习率调度器 (Cosine Annealing)
        lr = lr_cosine_schedule(step, args.max_learning_rate, args.min_learning_rate, args.warmup_iters, args.cosine_cycle_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 2. 评估验证集 & 保存 Checkpoint
        if step % args.eval_interval == 0:
            val_loss = estimate_loss(model, val_data, args.batch_size, args.context_length, 50, device)
            train_loss_est = estimate_loss(model, train_data, args.batch_size, args.context_length, 50, device)
            
            # 记录到 W&B (如果你用了的话)
            run.log({"eval/train_loss": train_loss_est, "eval/val_loss": val_loss}, step=step)
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     ckpt_path = os.path.join(args.out_dir, 'best_model.pt')
            #     save_checkpoint(model, optimizer, step, os.path.join(args.out_dir, ckpt_path))

        # 3. 定期全量保存 (防止断电)
        if step % args.save_interval == 0:
            ckpt_path = os.path.join(args.out_dir, f'{args.dataset}_{uid}_step{step}.pt')
            save_checkpoint(model, optimizer, step, ckpt_path)

        # 4. 执行一步训练
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        
        optimizer.zero_grad(set_to_none=True) # 性能优化：set_to_none 比 zero_ 更快
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()

        # 5. 梯度裁剪 (在 Optimizer Step 之前！)
        if args.max_l2_norm != 0.0: gradient_clipping(model.parameters(), args.max_l2_norm)
            
        if step % args.log_interval == 0:
            run.log({"train/loss": loss.item(), "train/lr": lr}, step=step)

        optimizer.step()


# --- 2. 验证集评估逻辑 ---
@torch.no_grad()
def estimate_loss(model, data, batch_size, context_length, eval_iters, device):
    """估算当前模型在数据集上的平均 Loss"""
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(data, batch_size, context_length, device)
        logits = model(x)
        # 这里的 logits 形状通常是 (B, T, V)，targets 是 (B, T)
        # 注意：如果你的 custom_cross_entropy 只处理 2D，这里需要 view(-1)
        loss = cross_entropy(logits, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()
                

# --- 3. 命令行参数解析 ---
def parse_args():
    parser = argparse.ArgumentParser(description="Tiny GPT Training Script")
    # 数据路径
    parser.add_argument("--data_dir", type=str, default="data/", help="保存 data 的目录")
    parser.add_argument("--out_dir", type=str, default="out/", help="保存 checkpoint 的目录")
    parser.add_argument("--dataset", type=str, default="TinyStories", help="要训练的数据集")
    # 模型超参 (与你的 GPT 类定义对应)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--theta", type=float, default=10000.0)
    parser.add_argument("--d_ff", type=int, default=1344)
    # 训练超参
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_iters", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--eval_interval", type=int, default=125)
    parser.add_argument("--save_interval", type=int, default=500)
    ## Optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    ## lr schedule
    parser.add_argument("--max_learning_rate", type=float, default=1e-2)
    parser.add_argument("--min_learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--cosine_cycle_iters", type=int, default=2500)
    ## gradient clipping
    parser.add_argument("--max_l2_norm", type=float, default=1.0)
    # 断点续传
    parser.add_argument("--resume", type=str, default=None, help="从某个 .pt 文件恢复训练")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_dict = vars(args).copy()
    ignore_keys = {"data_dir", "out_dir", "eval_interval", "save_interval"}
    for key in ignore_keys:
        config_dict.pop(key, None)

    uid = time.strftime('%m%d%H%M')
    params = dict(
        entity="rla-study",
        project="CS336",
        group="basics",
        job_type="training",
        name=f"{args.dataset}_{uid}",
        config=config_dict
    )
    device = 'cuda' if torch.cuda.is_available() else 'mps'
    with wandb.init(**params) as run:
        main(args, device, run, uid)