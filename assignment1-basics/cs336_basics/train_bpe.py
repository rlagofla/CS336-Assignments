'''
总览一下，最核心的函数放上面了，越次要越后面
2.5 的题目都挤在这了
'''

import multiprocessing
import regex as re
from collections import Counter, defaultdict
import os
from typing import BinaryIO
import psutil
import pickle
import cProfile
import pstats


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
input_path = 'data/TinyStoriesV2-GPT4-valid.txt'
vocab_size = 1000
special_tokens = ['<|endoftext|>']

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 初始化
    vocab = {idx:bytes([idx]) for idx in range(256)}
    idx = 256
    for st in special_tokens:
        vocab[idx] = st.encode('utf-8')
        idx += 1
    merges = []

    # 找边界，并行。只在预分词频率上并行
    with open(input_path, "rb") as f:
        num_processes = multiprocessing.cpu_count()
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    with multiprocessing.Pool(processes=num_processes) as pool:
        all_pre_cnt = pool.starmap(pre_tokenizer, zip(
            [input_path]*num_processes, 
            boundaries[:-1], boundaries[1:],
            [special_tokens]*num_processes))
    
    pre_cnt = Counter()
    for pc in all_pre_cnt:
        pre_cnt += pc
    # 相邻的频率
    adj_cnt = Counter()
    # cache 通过相邻找预分词
    # 还是字典，和 Counter 一样，方便用的
    adj_pre = defaultdict(list)
    
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

def pre_tokenizer(
    input_path: str,
    start: int,
    end: int,
    special_tokens: list[str]
):
    '''
    用来并行的函数

    Args:
        chunk 在文件中的起点
        chunk 在文件中的终点

    Returns:
        当前 chunk 的预分词频率
    '''

    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode('utf-8', errors='ignore')
    
    st_pat = '|'.join(map(re.escape, special_tokens))
    txts = re.split(st_pat, chunk)

    pre_cnt = Counter()
    for txt in txts:
        for pre in re.finditer(PAT, txt):
            tpl = tuple(bytes([b]) for b in pre.group().encode('utf-8'))
            pre_cnt[tpl] += 1

    return pre_cnt


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    官方的代码，没动

    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)  # 单位: GB

def save_tokenizer_assets(vocab, merges, vocab_path, merges_path):
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    with open(merges_path, 'wb') as f:
        pickle.dump(merges, f)


if __name__ == '__main__':
    # 创建性能分析器
    profiler = cProfile.Profile()
    profiler.enable() # 开始记录

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    profiler.disable() # 停止记录

    # 打印分析结果
    print(f"内存占用: {get_memory_usage():.2f} GB")
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(20)  # 只看前 20 名最慢的操作
    
    for i in sorted(vocab, reverse=True, key=lambda x: len(vocab[x]))[:5]:
        print(vocab[i])
    save_tokenizer_assets(vocab, merges, input_path[:-4]+'_vocab.pickle', input_path[:-4]+'_merges.pickle')