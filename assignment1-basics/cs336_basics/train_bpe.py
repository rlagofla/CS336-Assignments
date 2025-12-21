'''
用 git 管理一下

这里做一个简单尝试，先拿 valid 的 一个 chunk 做尝试 0, 5625758
'''

import regex as re
from collections import Counter, defaultdict
import sys

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 初始化
    with open(input_path, "rb") as f:
        chunk = f.read().decode('utf-8', errors='ignore')
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    vocab = {idx:bytes([idx]) for idx in range(256)}
    idx = 256
    for st in special_tokens:
        vocab[idx] = st.encode('utf-8')
        idx += 1

    merges = []
    
    # 拆分文字
    st_pat = '|'.join(map(re.escape, special_tokens))
    txts = re.split(st_pat, chunk)
    
    # 这里 Counter most common 给的是字典序升序
    # 预分词的频率计数
    pre_cnt = Counter()
    # 相邻的频率
    adj_cnt = Counter()

    # cache 通过相邻找预分词
    # 还是字典，和 Counter 一样，方便用的
    adj_pre = defaultdict(list)
    
    # 预分词并统计
    for txt in txts:
        for pre in re.finditer(PAT, txt):
            # 这样出来的是 整型元组：
            # pre_cnt[tuple(pre.group().encode('utf-8'))] += 1
            # 要 tuple 包一下，不然懒惰
            tpl = tuple(bytes([b]) for b in pre.group().encode('utf-8'))
            pre_cnt[tpl] += 1
    
    # 在预分词中统计相邻次数，并且构造 cache
    for k, v in pre_cnt.items():
        # 这里长度为 1 的会自动不满足循环
        for l, r in zip(k[:-1], k[1:]):
            adj_cnt[(l,r)] += v
            adj_pre[(l,r)].append(k)

    while len(vocab) < vocab_size:
        # 不用 most common，一个是默认字典升序，一个是不知道平局有多少个
        the_pair = max(adj_cnt.keys(), key=lambda p: (adj_cnt[p], p))
        # 字节 + 字节 = 字节拼接
        merged_pair = the_pair[0] + the_pair[1]
        # print(merged_pair)
        merges.append(the_pair)
        vocab[idx] = merged_pair
        idx += 1
        
        # 根据 cache adj_pre 更新频率
        # 清掉
        adj_cnt[the_pair] = 0
        for k in adj_pre[the_pair]:
            '''
            注意这里用到了 pre cnt，所以之后的更新仍然需要 pre cnt
            （我说白了就不要质疑人家的 pdf，人家说要更新 pre cnt 那就更新啊，干嘛这么倔）
            '''
            v = pre_cnt[k]
            if v == 0: continue   # 如果已经被 Lazy 删了，那就过
            # 草，还不能用 enumerate+zip，对于 aaa，zip 会出现两次合并
            new_k = []
            i = 0
            # 注意这里也还有坑，我们构造 new k 是要遍历完整个 k 的，我这样会少遍历一个...
            while i < len(k):
                if i >= len(k)-1 or (k[i],k[i+1]) != the_pair:
                    new_k.append(k[i])
                    i += 1
                    continue
                # 相等
                new_k.append(merged_pair)
                if i != 0:
                    adj_cnt[(k[i-1], merged_pair)] += v
                    adj_cnt[(k[i-1], the_pair[0])] -= v
                if i != len(k)-2:
                    adj_cnt[(merged_pair, k[i+2])] += v
                    adj_cnt[(the_pair[1], k[i+2])] -= v
                i += 2
            new_k = tuple(new_k)
            pre_cnt[k] = 0  # Lazy 删一下
            pre_cnt[new_k] += v
            '''
            所以 adj_pre 怎么更新？想不到一个好的那就直接暴力更新，就模仿前面的对 new k 更新一下
            '''
            for l, r in zip(new_k[:-1], new_k[1:]):
                adj_pre[(l,r)].append(new_k)

    
    return vocab, merges

train_bpe('data/TinyStoriesV2-GPT4-valid.txt', 500, ['<|endoftext|>'])