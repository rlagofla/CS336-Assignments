import numpy as np
import os
import multiprocessing
from tqdm import tqdm
import shutil
from .tokenizer import Tokenizer

# --- 复用你之前的边界查找函数 ---
def find_chunk_boundaries(input_path, desired_num_chunks):
    """
    (简化版逻辑) 简单的按字节均分，但要确保不切断行。
    这里假设每一行都是独立的 doc，或者切断 doc 也没关系（因为有 encoder 处理）。
    为了安全，我们通常按换行符 '\n' 对齐。
    """
    with open(input_path, 'rb') as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        
    chunk_size = file_size // desired_num_chunks
    boundaries = [0]
    
    with open(input_path, 'rb') as f:
        for i in range(1, desired_num_chunks):
            target_pos = i * chunk_size
            f.seek(target_pos)
            # 读到下一个换行符，确保边界在行尾
            f.readline()
            boundaries.append(f.tell())
    
    boundaries.append(file_size)
    return sorted(list(set(boundaries)))

# --- Worker 函数：每个进程独立干活 ---
def worker_process(args):
    """
    每个 Worker 负责处理文件的一部分，并写入独立的临时文件
    """
    input_path, start, end, tokenizer, temp_output_path, worker_id = args
    
    # print(f"Worker {worker_id} starting: {start} -> {end}")
    
    tokens_buffer = []
    CHUNK_SIZE = 1024 * 1024 # 内存缓冲大小
    
    with open(input_path, 'r', encoding='utf-8') as f:
        f.seek(start)
        
        # 打开临时文件准备写入
        with open(temp_output_path, 'wb') as f_out:
            while f.tell() < end:
                line = f.readline()
                if not line: break # EOF
                
                # 编码
                # 注意：这里假设 tokenizer.encode 处理单行。
                # 如果你的 tokenizer 有特殊的 encode_iterable，也可以在这里改造成小批量的 iterable
                ids = tokenizer.encode(line) 
                tokens_buffer.extend(ids)
                
                # 缓冲区满了就写入磁盘
                if len(tokens_buffer) > CHUNK_SIZE:
                    arr = np.array(tokens_buffer, dtype=np.uint16)
                    f_out.write(arr.tobytes())
                    tokens_buffer = []
            
            # 写入剩余的
            if tokens_buffer:
                arr = np.array(tokens_buffer, dtype=np.uint16)
                f_out.write(arr.tobytes())
                
    return temp_output_path

# --- 主函数 ---
def process_dataset_parallel(tokenizer, input_file_path, output_file_path, num_workers=None):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()  # 留一个核给系统? 也可以 max
    
    print(f"Parallel processing {input_file_path} with {num_workers} workers...")
    
    # 1. 划分边界
    boundaries = find_chunk_boundaries(input_file_path, num_workers)
    # 实际切出的块数可能少于 num_workers (如果文件很小)
    real_chunks = len(boundaries) - 1
    
    # 2. 准备任务参数
    tasks = []
    temp_files = []
    
    # 确保临时目录存在
    os.makedirs("tmp_shards", exist_ok=True)
    
    for i in range(real_chunks):
        start = boundaries[i]
        end = boundaries[i+1]
        temp_path = f"tmp_shards/{os.path.basename(input_file_path)}.part{i}.bin"
        temp_files.append(temp_path)
        
        # 注意：这里直接传 tokenizer 对象。如果对象很大（比如包含巨大 dict），
        # 可能导致序列化开销。通常 BPE 的 dict 也就几 MB，问题不大。
        tasks.append((input_file_path, start, end, tokenizer, temp_path, i))

    # 3. 并行执行
    with multiprocessing.Pool(processes=num_workers) as pool:
        # 使用 imap_unordered + tqdm 显示进度
        # 注意：这里进度条显示的是“完成了几个 Chunk”，而不是具体的行数
        for _ in tqdm(pool.imap_unordered(worker_process, tasks), total=len(tasks), desc="Processing Chunks"):
            pass

    # 4. 合并文件 (Gather)
    print("Merging shards...")
    with open(output_file_path, 'wb') as outfile:
        for temp_path in temp_files:
            if os.path.exists(temp_path):
                with open(temp_path, 'rb') as infile:
                    # 使用 shutil.copyfileobj 高效利用系统调用进行拷贝
                    shutil.copyfileobj(infile, outfile)
                os.remove(temp_path) # 删掉临时文件
    
    # 清理目录
    if not os.listdir("tmp_shards"):
        os.rmdir("tmp_shards")

    print(f"Done! Saved to {output_file_path}")

# --- 使用示例 ---
if __name__ == '__main__':
    # 必须放在 if __name__ == '__main__' 下面，否则 Windows/Mac 会报错
    tokenizer = Tokenizer.from_files("data/TinyStoriesV2-GPT4-valid_vocab.pickle", "data/TinyStoriesV2-GPT4-valid_merges.pickle", ['<|endoftext|>'])
    
    input_file = "data/TinyStoriesV2-GPT4-valid.txt"
    output_file = "data/TinyStoriesV2-GPT4-valid.bin"
    
    process_dataset_parallel(tokenizer, input_file, output_file)