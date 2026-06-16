# FastWAM 接入 C1/C2 方案分析

## 核心结论

**不需要任何数据格式转换。** 直接用 FastWAM 预处理好的 RoboTwin 数据，通过 episode 过滤实现 C1/C2 split。

## 为什么"谁转谁"的问题答案是"都不转"

| 方案 | 可行性 | 原因 |
|------|--------|------|
| 我们的 JSON → FastWAM Lerobot | ❌ 不可行 | 原始视频帧不在本地（`/mnt/nas/mixiangju/Data/RoboTwin/` 不可访问） |
| FastWAM 读我们的 JSON | ❌ 不可行 | FastWAM 需要 33 帧连续视频，我们只有单帧 keyframe |
| **直接用 FastWAM 预处理数据 + 过滤** | ✅ **最现实** | 下载他们处理好的数据，按 episode 过滤 |

## 数据关联方式

我们的 C2 JSON 中每个 frame 有 `source_episode_dir`：
```
/mnt/nas/mixiangju/Data/RoboTwin/C2-v0/train/C2_move_block/c2_pilot_train_move_block_blue_large_seed_407008
```

FastWAM 预处理数据中的 episode 命名也基于原始 RoboTwin episode ID。通过 `seed_{X}` 可以建立一一映射。

## 实施步骤

### Step 1: 下载 FastWAM 预处理数据（~1-2 小时）

```bash
# 在 FastWAM 目录
mkdir -p data/robotwin2.0
cd data/robotwin2.0
# 下载 yuanty/robotwin2.0-fastwam 所有 part 文件
# cat robotwin2.0.tar.gz.part-* | tar -xzf -
```

### Step 2: 生成 episode 过滤索引（~30 分钟）

从我们的 C2 train JSON 提取 episode ID 列表，生成 `c2_train_episode_ids.txt`。

### Step 3: 修改 FastWAM dataset 支持过滤（~2-3 小时）

在 `BaseLerobotDataset.__init__` 中增加 `episode_filter_path` 参数：
- 如果提供了过滤列表，只加载列表中的 episode
- 否则加载全部

### Step 4: 预计算 T5 text embedding（~30 分钟）

```bash
python scripts/precompute_text_embeds.py task=robotwin_uncond_3cam_384_1e-4
```

### Step 5: 训练（8 GPU，~1-2 天）

```bash
bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4
# 需要覆盖 data config，指向过滤后的数据集
```

## 工作量总结

| 步骤 | 时间 | 难度 |
|------|------|------|
| 下载数据 | 1-2h | 低 |
| 写 episode filter | 2-3h | 中 |
| 修改 dataset | 2-3h | 中 |
| 预计算 embedding | 30min | 低 |
| 训练（8 GPU） | 1-2 天 | 低 |
| **总计** | **~2 天** | **中** |

## 与"从头训练我们的数据"对比

| 对比项 | 本方案 | 从头转换 |
|--------|--------|----------|
| 工作量 | ~2 天 | ~2 周 |
| 数据完整性 | 完整 33 帧 + 3 camera | 只有单帧/2 camera |
| 视频质量 | FastWAM 预处理（高质量） | 需要重新编码 |
| 可靠性 | 高（用官方预处理） | 低（自定义转换易出错） |

## 下一步

如果你同意这个方案，我可以立刻开始：
1. 写 episode filter 脚本
2. 修改 FastWAM dataset 代码
3. 写 1 GPU 测试脚本（验证过滤逻辑正确）
