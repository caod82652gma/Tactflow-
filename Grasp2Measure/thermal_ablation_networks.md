# Grasp2Measure 训练说明

## 介绍

### 数据

原始热-触觉数据位于：

`Workspace/C_model_training/thermal/batch_*_*ml`

原始 `.bin` 是连续采集序列。`Workspace/C_model_training/thermal/scripts/process_shape_data.py` 先解析 `.bin`，扣除零点基线，并输出逐帧 merged CSV：

`Workspace/C_model_training/thermal/data/merged`

静态液位分类使用 merged CSV 的最后 `steady_tail_frames=8` 帧均值，表示接触后稳定状态。

时序液位分类使用：

`Workspace/C_model_training/thermal/scripts/generate_thermal_sequences.py`

从 merged CSV 中检测接触开始帧，截取接触后窗口，做 MAD 异常值裁剪，并重采样/补齐为定长序列，默认 `sequence_length=128`。输出为：

`Workspace/C_model_training/thermal/data/sequence`

### 物体分类

物体分类是触觉容器分类任务，主要用于预测 `container_one_hot`，供液位消融分支使用。

常用输入：

- 触觉稳定帧摘要
- 16 维压力特征 `pressure16`

### 液位分类

液位类别：

`Low, Mid, High`

静态液位分类包含 B0-B6：

| 编号 | 名称 | 输入特征 | 模型 |
|---|---|---|---|
| B0 | `b0_physics` | `DeltaT`, `T_amb`, `T_liquid` | 物理公式 |
| B1 | `b1_temp_only` | `DeltaT` | MLP |
| B2 | `b2_weighted_temp_only` | `alpha * DeltaT` | MLP |
| B3 | `b3_no_container_one_hot` | `alpha * DeltaT`, `alpha`, `tau_hat` | MLP |
| B4 | `b4_no_tau_hat` | `alpha * DeltaT`, `alpha`, `container_one_hot` | MLP |
| B5 | `b5_full_fusion` | `alpha * DeltaT`, `alpha`, `container_one_hot`, `tau_hat` | MLP |
| B6 | `b6_direct_pressure16_temp` | `pressure16`, `DeltaT` | MLP |

静态 B1-B6 网络：

`Linear(input_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, 3)`

时序 B1-B6 保持同样的消融输入组合，只把输入从 `[N, F]` 改为 `[N, T, F]`：

`[B, T, F] -> encoder -> Linear(32) -> ReLU -> Linear(3)`

支持 encoder：

- `gru`
- `lstm`
- `cnn`
- `transformer`

## 使用示例

以下命令默认在项目根目录运行：

`C:\Users\FangYuxuan\Saved Games\Upper_VET6USB`

推荐使用 conda 环境 Python：

`C:\Users\FangYuxuan\anaconda3\envs\vet6usb_pyqt\python.exe`

### 1. 物体分类

基础训练：

```powershell
cd Grasp2Measure
C:\Users\FangYuxuan\anaconda3\envs\vet6usb_pyqt\python.exe train.py --config configs/default.yaml
```

5-fold：

```powershell
cd Grasp2Measure
C:\Users\FangYuxuan\anaconda3\envs\vet6usb_pyqt\python.exe train.py --config configs/default.yaml --cv-folds 5
```

如果只跑某一折：

```powershell
cd Grasp2Measure
C:\Users\FangYuxuan\anaconda3\envs\vet6usb_pyqt\python.exe train.py --config configs/default.yaml --cv-folds 5 --fold 0
```

### 2. 静态液位分类

基础命令：

```powershell
cd Grasp2Measure
C:\Users\FangYuxuan\anaconda3\envs\vet6usb_pyqt\python.exe thermal_train.py --config configs/thermal.yaml
```

常用 5-fold 示例：

```powershell
cd Grasp2Measure
C:\Users\FangYuxuan\anaconda3\envs\vet6usb_pyqt\python.exe thermal_train.py --config configs/thermal.yaml --epochs 300 --batch-size 16 --lr 0.0003 --output-dir runs/thermal_cv5 --cv-folds 5
```

只跑第 0 折：

```powershell
cd Grasp2Measure
C:\Users\FangYuxuan\anaconda3\envs\vet6usb_pyqt\python.exe thermal_train.py --config configs/thermal.yaml --epochs 300 --batch-size 16 --lr 0.0003 --output-dir runs/thermal_cv5 --cv-folds 5 --fold 0
```

### 3. 生成时序液位数据

基础生成：

```powershell
C:\Users\FangYuxuan\anaconda3\envs\vet6usb_pyqt\python.exe Workspace\C_model_training\thermal\scripts\generate_thermal_sequences.py
```

指定序列长度：

```powershell
C:\Users\FangYuxuan\anaconda3\envs\vet6usb_pyqt\python.exe Workspace\C_model_training\thermal\scripts\generate_thermal_sequences.py --sequence-length 128
```

### 4. 时序液位分类

基础命令：

```powershell
cd Grasp2Measure
C:\Users\FangYuxuan\anaconda3\envs\vet6usb_pyqt\python.exe thermal_sequence_train.py --config configs/thermal_sequence.yaml --encoder gru
```

5-fold GRU：

```powershell
cd Grasp2Measure
C:\Users\FangYuxuan\anaconda3\envs\vet6usb_pyqt\python.exe thermal_sequence_train.py --config configs/thermal_sequence.yaml --encoder gru --epochs 300 --batch-size 16 --lr 0.0003 --cv-folds 5
```

只跑第 0 折：

```powershell
cd Grasp2Measure
C:\Users\FangYuxuan\anaconda3\envs\vet6usb_pyqt\python.exe thermal_sequence_train.py --config configs/thermal_sequence.yaml --encoder gru --epochs 300 --batch-size 16 --lr 0.0003 --cv-folds 5 --fold 0
```

切换 encoder：

```powershell
cd Grasp2Measure
C:\Users\FangYuxuan\anaconda3\envs\vet6usb_pyqt\python.exe thermal_sequence_train.py --config configs/thermal_sequence.yaml --encoder cnn --cv-folds 5
```

指定输出目录：

```powershell
cd Grasp2Measure
C:\Users\FangYuxuan\anaconda3\envs\vet6usb_pyqt\python.exe thermal_sequence_train.py --config configs/thermal_sequence.yaml --encoder transformer --output-dir runs/thermal_sequence_transformer_cv5 --cv-folds 5
```

指定序列长度：

```powershell
cd Grasp2Measure
C:\Users\FangYuxuan\anaconda3\envs\vet6usb_pyqt\python.exe thermal_sequence_train.py --config configs/thermal_sequence.yaml --encoder gru --sequence-length 128 --cv-folds 5
```

说明：训练时的 `--sequence-length` 应与生成时序数据时使用的长度一致。

补生成 summary 文件夹：

```powershell
cd Grasp2Measure
C:\Users\FangYuxuan\anaconda3\envs\vet6usb_pyqt\python.exe thermal_sequence_train.py --config configs/thermal_sequence.yaml --encoder gru --summarize-only
```
