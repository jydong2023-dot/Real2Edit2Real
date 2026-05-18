import os
from huggingface_hub import snapshot_download

# 启用 hf_transfer 可以大幅提升大模型的下载速度（可选，但强烈推荐）
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

repo_id = "nvidia/Cosmos-Predict2-2B-Video2World"
local_dir = "checkpoints/Cosmos-Predict2-2B-Video2World"

print(f"开始下载/续传模型: {repo_id}")
print(f"目标文件夹: {local_dir}")

try:
    # snapshot_download 默认自带断点续传机制
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False, # 设为 False 表示直接将文件下载到目标目录，而不是使用软链接
        resume_download=True,         # 显式声明断点续传（尽管新版本底层默认开启）
        max_workers=8                 # 多线程下载，提升速度
    )
    print("下载完成！")
except Exception as e:
    print(f"下载被中断或发生错误: {e}")
    print("重新运行此脚本即可从断点继续下载。")