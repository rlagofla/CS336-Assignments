import torch
from ..ch3.softmax import softmax
from ..ch2.tokenizer import Tokenizer

@torch.no_grad()
def generate(model, tokenizer: Tokenizer, prompt, max_new_tokens, temperature=1.0, top_p=None, eos_token=None):
    """
    GPT 生成函数
    
    Args:
        model: 训练好的 GPT 模型
        prompt: 
        max_new_tokens: 最多生成多少个新 token
        temperature: 温度缩放 (1.0 表示不缩放, >1.0 更随机, <1.0 更保守)
        top_p: 核采样阈值 (None 或 1.0 表示不使用)
        eos_token_id: 终止符 ID，如果生成了它就提前停止
    """
    model.eval()

    eos_token_id = None
    if eos_token is not None:
        eos_token_id = tokenizer.encode(eos_token)[0]
    device = next(model.parameters()).device

    idx = tokenizer.encode(prompt)
    # 居然是无脑 (S,) -> (1, S) 吗
    idx = torch.tensor(idx, device=device, dtype=torch.long).unsqueeze(0)
    
    for _ in range(max_new_tokens):
        # 居然是无脑截断吗
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        
        logits = model(idx_cond)
        # 只取最后一个词的概率分布，配合温度
        logits = logits[:, -1, :] / temperature
        # top-p
        if top_p is not None and top_p < 1.0:
            logits = top_p_filtering(logits, top_p)
        
        probs = softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        # 判断终止
        if eos_token_id is not None and (idx_next == eos_token_id).all():
            break
            
    return tokenizer.decode(idx.squeeze(0))


def top_p_filtering(logits: torch.Tensor, top_p):
    # 排序还能拿索引
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)
    
    # 3. 构建排序空间下的 sorted_indices_to_remove
    # [False, False, True, True] 删掉 True，但是第一个 True 要保留
    sorted_indices_to_remove = cumulative_probs > top_p
    
    # 向右移动 sorted_indices_to_remove 以保留“正好超过阈值”的那一个词
    # [False, False, True, True] -> [False, False, False, True]
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    # 边界条件，防止全都 > top_p
    sorted_indices_to_remove[..., 0] = False

    # scatter 类似于 gather，然后这里比之前学的 gather 复杂多了
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    
    logits = logits.masked_fill(mask, float('-inf'))
    return logits

if __name__ == "__main__":
    pass