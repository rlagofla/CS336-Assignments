import pickle
from typing import Iterable, Iterator
import regex as re
import heapq
import os


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer():
    def __init__(self, 
            vocab: dict[int, bytes], 
            merges: list[tuple[bytes, bytes]], 
            special_tokens=None
        ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []

        # 把用户提供的 special tokens 加入进来
        max_id = len(vocab)
        for st in self.special_tokens:
            st_bytes = st.encode('utf-8')
            if st_bytes not in vocab.values():
                max_id += 1
                self.vocab[max_id] = st_bytes

        self.inv_vocab = {v:k for k, v in vocab.items()}
        self.inv_merges = {merges[i]:i for i in range(len(merges))}
        
        # special_tokens 的 split 规则
        self.st_pat = None
        if self.special_tokens:
            # 正则表达式有匹配顺序，优先大的匹配
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            # 谁知道啊，加上这个括号就能保留分割了
            self.st_pat = re.compile('(' + '|'.join(map(re.escape, sorted_special_tokens)) + ')')

    # 工厂函数
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        out = []
        # 一定要这么搞一下因为 '()' 会把每个字符直接隔开
        if self.st_pat:
            chunks = self.st_pat.split(text)
        else:
            chunks = [text]

        for chunk in chunks:
            if chunk in self.special_tokens:
                out.append(self.inv_vocab[chunk.encode('utf-8')])
                continue
            for pre in re.finditer(PAT, chunk):
                lst_idx = self.merge(pre.group())
                out.extend(lst_idx)
        return out


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for it in iterable:
            for idx in self.encode(it):
                yield idx


    def decode(self, ids: list[int]) -> str:
        out = b''
        # 不能一个数字一个数字的 decode，要最后一起 decode
        for idx in ids:
            out += self.vocab[idx]
        # 忘记了，pdf 这里说了，希望你处理不了的地方用 replace    
        return out.decode('utf-8', errors='replace')


    def merge(self, pre: str) -> list[int]:
        '''
        合并最复杂，用小根堆还没结束，后面怎么合并很晕
        train_bpe 那边，是直接摆烂，用 lazy 删拒绝合并
        这里用双向链表，这样不用动表结构，改改指针就合并（可能会问，为什么要双向，因为合并后也要看看前驱有没有可能合并加入堆）
        '''
        pre_bytes = pre.encode('utf-8')

        # 用二维数组代替双向链表
        # -1 代表 Null
        linked = []
        for i, b in enumerate(pre_bytes):
            # node 结构：val, prev, next, isVaild（被越过的就是无效）
            linked.append([bytes([b]), i-1, i+1, True])
        linked[-1][2] = -1


        min_heap = []
        def push_pair(l, r):
            # 后面追加的时候特判：
            if l == -1 or r == -1:
                return
            pair = (linked[l][0], linked[r][0])
            if pair in self.inv_merges:
                # 放进去的是 merge 的顺序 和 merge 在 pre 中发生的位置
                heapq.heappush(min_heap, (self.inv_merges[pair], l))

        # 一开始都是两两相邻
        for i in range(len(linked) - 1):
            push_pair(i, i+1)
        

        while min_heap:
            rank, i = heapq.heappop(min_heap)            
            '''
            合并之前，还是要 lazy check 的。考虑 hello 如果查表发现 ll lo 都能合并
            那 ll 合并后 lo 就失效了，所以要 check
            '''
            if not linked[i][3]: continue

            '''比如 the，全都能合并，t h e 先合并 t he，然后 the，但是堆里有 (t h)，和 (t he) 合并都是一样的 i nexti'''
            # 找下一个
            next_i = linked[i][2]
            curr_pair = (linked[i][0], linked[next_i][0])
            if self.inv_merges.get(curr_pair, -1) != rank: continue

            # 更新合并值
            linked[i][0] += linked[next_i][0]
            # 被跳过的取消掉
            linked[next_i][3] = False
            # 拿出真正的下一个，改前驱后继
            n_next_i = linked[next_i][2]
            if n_next_i != -1: linked[n_next_i][1] = i
            linked[i][2] = n_next_i

            # 合并后，会出现新的两对相邻
            # 注意合并后可能出现在开头和结尾，所以要特判，就不放在这判断了，放函数里头
            push_pair(linked[i][1], i)
            push_pair(i, linked[i][2])

        out = []
        i = 0
        while i != -1:
            out.append(self.inv_vocab[linked[i][0]])
            i = linked[i][2]
        return out

if __name__ == '__main__':
    tokenizer = Tokenizer.from_files('data/TinyStoriesV2-GPT4-valid_vocab.pickle', 'data/TinyStoriesV2-GPT4-valid_merges.pickle', ['<|endoftext|>'])
