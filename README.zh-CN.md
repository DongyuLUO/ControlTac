# ControlTac

[English README](README.md) · [论文](https://arxiv.org/abs/2505.20498) · [恢复说明](docs/RECOVERY.md)

这是从实验代码恢复并整理的独立发布目录。原始实验文件没有移动或删除。两个 checkpoint 的模型张量保持完全一致；数据划分和训练入口是重新构建的，不代表已经重新复现论文全部指标。

## 快速开始

在这个目录下执行，Python 建议使用 3.11。先安装与 CUDA 匹配的 PyTorch/torchvision，再安装项目：

```bash
pip install -e .
python run.py infer --stage force
python run.py infer --stage force_pose
```

示例输入已经放在 `examples/`，输出在 `outputs/`。首次运行会下载 DC-AE。已经缓存时可加 `--local-only`。没有 GPU 时加 `--device cpu`；显存充足时可以加 `--codec-device cuda` 加速编码与解码。

## 权重

| 旧文件 | 新文件 |
| --- | --- |
| force_control.pth | checkpoints/force_control.pth |
| force_pose_control.pth | checkpoints/force_pose_control.pth |

新文件只包含模型参数与模型必要 buffer，不包含 epoch、optimizer、scheduler 或 scaler。文件大小分别约 590 MB、892 MB。原始 checkpoint 保留；逐张量一致性和 SHA-256 见 `checkpoints/manifest.json`。

发布用纯权重位于 `ControlTac/checkpoints/`，按作者要求使用原文件名；实验目录 `FT/` 根目录下同名文件仍是完整训练 checkpoint，注意区别目录。

所有物体、两个阶段统一使用 `shared` 归一化配置，默认自动选择，无需传入物体名或归一化 key。`--normalization` 用于指定保存的配置文件。

## 数据划分

六个物体统一命名为 Cross、Slim Cylinder、Thin Cylinder、Medium Cylinder、Big Sphere 和 Triple Cylinder。

第一阶段严格为 20,000 个 samples，第二阶段为 7,000 张不重复图片。每个物理物体的份额尽可能均衡，Thin Cylinder 的全部来源记录合计占一个物体的份额。不能整除造成的余数以固定方式分配。

**实际缺口：** 为保留 Thin Cylinder 原定的来源采样比例，其中一组来源保留独立验证、测试姿态后可用图片不足。当前恢复方案固定重复采样 176 次，因此第一阶段是 **19,824 张不同图片、20,000 个 samples**。这不是新增的真实图片，不能据此宣称拥有 20,000 张 unique images。可以关闭重复采样重新生成，但此时脚本会明确报数量不足。

第二阶段六个物体各有 300 个不同姿态；Thin Cylinder 合并统计为一个物体，共 300 个不同姿态。训练、验证、测试之间没有姿态交叉。验证集 911 条，测试集 957 条。原始 CSV、每条样本来源和统计见 `splits/`。

## 训练

当前工作区的数据根目录为本目录的上一级：

```bash
python tools/verify_data.py --data-root ..
python run.py train --data-root ..
```

该命令先训练 force，再用生成的权重初始化 force+pose。默认采用论文的 75,000 steps / 阶段、batch size 4、AdamW、cosine 学习率和等权 L1/MSE。仅做快速测试可以加 `--max-steps 1 --device cpu --local-only`。

两阶段默认保存纯权重及独立配置、归一化 JSON、loss 日志。需要断点续训时，使用单阶段入口加 `--save-training-state`；恢复时传 `--resume`。具体命令见英文 README。

重建数据划分：

```bash
python -m controltac.prepare --data-root .. --allow-repeated-force-samples
```

直接使用 [FeelAnyForce](https://huggingface.co/datasets/amirsh1376/FeelAnyForce) 已有的扣背景图片，不重复发布图像数据包。我们只在 [private Release](https://github.com/DongyuLUO/ControlTac/releases/tag/v0.1.0) 提供权重及小型 `controltac_annotations.zip`（对齐 mask 和图像校验索引）。下载需要登录拥有仓库访问权限的 GitHub 账号。

解压 FeelAnyForce 后执行：

```bash
python tools/link_feelanyforce.py --source-root /path/to/FeelAnyForce --annotations controltac_annotations.zip --output data
python tools/verify_data.py --data-root data
python run.py train --data-root data
```

脚本逐张验证图片像素与训练清单一致，再建立本地链接；同一磁盘无需复制图像。跨盘无法链接时可加 `--copy`。不会把未扣背景的图片静默替换进去。详情见 [docs/DATA.md](docs/DATA.md)。

## 已验证和限制

已运行两个真实权重的推理、两个阶段各一次包含 16 对样本的 optimizer step，以及数据统计、姿态隔离、DDIM 和归一化检查。没有在本机重新训练完整的 150,000 steps，也没有声称复现论文指标。

旧 ControlNet 权重保留旧模型跳过第 0 个 backbone block 的行为；新训练路径修正该问题。因此使用新训练权重推理时要同时传训练产生的 `--model-config` 和 `--normalization`。所有物体和两个阶段均使用维护者确认的同一组共享归一化参数。

代码的上游授权来源已记录于 `THIRD_PARTY_NOTICES.md`。原目录没有项目 LICENSE，发布前仍需作者确认来源并选择适用的项目许可证。
