'''
总览一下，最核心的函数放上面了，越次要越后面
2.5 的题目都挤在这了

owt 跑了 10h 我去，看了一下主要是后面的合并花太久了，具体是 max 那一步
可以用大顶堆优化。由于堆的实现和一般字典之间的差异，两者要打配合：

功能              , 字典 (adj_cnt)  , 堆 (max_heap)
查询特定元素的当前值 , O(1) (极其快)   , O(N) (需要遍历整个树)
修改特定元素的值    , O(1)           , O(N) (找元素) + O(logN) (调整)
获取当前全局最大值  , O(N) (需全表扫描), O(1) (直接看堆顶)
插入新值          , O(1)           , O(logN) (调整) 

找最大值原先的 max 方案不行，但是堆快；但找到最大值之后要合并更新，更新字典快，堆慢。所以二者打配合
为了保证堆和字典之间的数据一致，可以用 lazy 删(O(1)) 和 插入新值来权衡

所以具体大顶堆的优化思路是：
1. 额外准备一个堆
2. 初始化堆
3. 每次用堆取出最大的都判断一下是不是和字典里的一致
4. 后面更新字典的时候一起更新
    - 插入新的，堆要插入
    - 已有的增加或减少，堆也选择插入（前面的判断一致就是 lazy 删）

最后，有关 bpe 的稳定，频率一致时，需要字典序高的，但是堆默认是降的，所以得自定义
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
import heapq


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
input_path = 'data/TinyStories.txt'
vocab_size = 1024
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

    # 新建堆，初始化堆
    max_heap = []
    for k, v in adj_cnt.items():
        heapq.heappush(max_heap, PairWrapper((v, k)))

    while len(vocab) < vocab_size:
        while True:  # 这边就是判断 lazy 删
            wrapped_pair = heapq.heappop(max_heap)
            cnt, the_pair = wrapped_pair.get()
            if cnt == adj_cnt[the_pair]:
                break
        
        # 字节 + 字节 = 字节拼接
        merged_pair = the_pair[0] + the_pair[1]
        # print(merged_pair)
        merges.append(the_pair)
        vocab[idx] = merged_pair
        idx += 1
        
        # 根据 cache adj_pre 更新频率
        # 清掉
        def update_adj_cnt_and_heap(pair, delta):
            adj_cnt[pair] += delta
            heapq.heappush(max_heap, PairWrapper((adj_cnt[pair], pair)))
        adj_cnt[the_pair] = 0
        # 都说堆和字典要同步，这里不用是因为，这里字典也是 lazy 删
        # 堆的 lazy 删靠的是不一致，那一边修改确实不一致
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
                    update_adj_cnt_and_heap((k[i-1], merged_pair), v)
                    update_adj_cnt_and_heap((k[i-1], the_pair[0]), -v)
                if i != len(k)-2:
                    update_adj_cnt_and_heap((merged_pair, k[i+2]), v)
                    update_adj_cnt_and_heap((the_pair[1], k[i+2]), -v)
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


class PairWrapper:
    def __init__(self, pair):
        self.pair = pair
        
    def __lt__(self, other):
        return self.pair > other.pair
    
    def __eq__(self, other):
        return self.pair == other.pair
        
    def __repr__(self):
        return str(self.pair)
    
    def get(self):
        return self.pair
    

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