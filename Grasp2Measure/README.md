# Grasp2Measure

Grasp2Measure trains grasp-based models for container recognition and, once the
next dataset is collected, thermal liquid-level estimation.

Current implemented pipeline:

- tactile container classifier: one CSV file is treated as one grasp trial.
  The classifier summarizes all frames in that file and predicts the container
  model: `15ml`, `50ml`, `100ml`, or `200ml`.

Default data:

```text
Workspace/C_model_training/tactile/idw4/{15ml,50ml,100ml,200ml}/*.csv       train/validation
Workspace/C_model_training/tactile_test/idw4/{15ml,50ml,100ml,200ml}/*.csv  final test
```

Train:

```bash
cd Grasp2Measure
python -m pip install -r requirements.txt
python train.py --config configs/default.yaml
```

Current completed runs:

```text
runs/tactile_base        test_acc=0.9333  test_loss=0.6170
runs/tactile_mean        test_acc=0.9143  test_loss=0.5308
runs/tactile_idw4        test_acc=0.9429  test_loss=0.3647
runs/tactile_idw4_smoke  test_acc=0.3048  test_loss=1.3608
```

If you use the `vet6usb_pyqt` conda environment on Windows, run the environment
Python directly:

```powershell
C:\Users\FangYuxuan\anaconda3\envs\vet6usb_pyqt\python.exe -m pip install -r requirements.txt
C:\Users\FangYuxuan\anaconda3\envs\vet6usb_pyqt\python.exe train.py --config configs/default.yaml
```

Evaluate:

```bash
cd Grasp2Measure
python eval.py --checkpoint runs/default/best.pt
```

Train the tactile models. Add `--eval` when you want `eval.py` to run automatically
after training; both training and evaluation write failed test samples to
`failure.csv` in the run output directory:

```powershell
conda activate vet6usb_pyqt
cd "C:\Users\FangYuxuan\Saved Games\Upper_VET6USB\Grasp2Measure"

python -m pip install -r requirements.txt

# 纯触觉 Summary 分支
python train.py --config configs/default.yaml --interpolation mean --input-mode tactile_summary --output-dir runs/tactile_mean_summary --eval

python train.py --config configs/default.yaml --interpolation base --input-mode tactile_summary --output-dir runs/tactile_base_summary --eval

python train.py --config configs/default.yaml --interpolation idw4 --input-mode tactile_summary --output-dir runs/tactile_idw4_summary --eval

# 纯触觉 16dim 分支
python train.py --config configs/default.yaml --interpolation idw4 --input-mode pressure_16 --output-dir runs/tactile_idw4_16dim --eval

# 生成中文图片
python Grasp2Measure\scripts\generate_cn_confusion_heatmaps.py

# 添加 wandb
--wandb --run-name tactile_idw4_summary

```


Evaluate a saved model and write the test confusion heatmap:

```powershell
python eval.py --checkpoint runs/tactile_base_summary/best.pt

python eval.py --checkpoint runs/tactile_idw4_summary/best.pt

python eval.py --checkpoint runs/tactile_idw4_16dim/best.pt
```

Temp Command-line overrides:

```powershell
conda activate vet6usb_pyqt
cd "C:\Users\FangYuxuan\Saved Games\Upper_VET6USB\Grasp2Measure"

C:\Users\FangYuxuan\anaconda3\envs\vet6usb_pyqt\python.exe thermal_train.py --config configs/thermal.yaml --epochs 300 --batch-size 16 --lr 0.0003 --output-dir runs/thermal_ablation_lr3e4
```


Useful hyperparameters:

- `--epochs`: maximum training epochs. Increase when train and validation loss are still going down; decrease for smoke tests.
- `--batch-size`: mini-batch size. Larger batches are faster and smoother when memory allows; smaller batches add noise and may generalize better.
- `--lr`: AdamW learning rate. If validation loss jumps around or diverges, try `0.0003`; if learning is too slow, try `0.003`.
- `model.hidden_dim` in `configs/default.yaml`: base hidden width. The training script multiplies this by 8 for the tactile MLP, so `16` becomes `128`.
- `model.dropout`: regularization. Raise it, for example from `0.1` to `0.2`, when train accuracy is high but validation/test accuracy is worse.
- `train.weight_decay`: AdamW L2 regularization. Try `0.0003` or `0.001` when overfitting is obvious.
- `features.tactile_summary`: currently `stable_frame`. For each CSV trial,
  the loader finds the adjacent frame pair with the smallest normalized
  multi-channel motion and uses the first frame in that pair. This removes
  obvious contact shaking before both training and PCA clustering. Alternatives
  are `stable_pair_mean`, `mean`, and `mean_std`.
- `data.val_ratio`, `data.random_seed`: control the deterministic train/validation split inside `data.tactile_root`.
- `data.tactile_test_root`: independent tactile test set used after training and by `eval.py`.

Dataset clustering and bad-data inspection:

```powershell
python scripts/cluster_tactile_trials.py --config configs/default.yaml --interpolation idw4 --output-dir runs/cluster_tactile
```

This writes:

```text
runs/cluster_tactile/idw4_cluster_pca.png
runs/cluster_tactile/idw4_cluster_report.csv
```

The plot shows the cleaned trial-level tactile features after standardization
and PCA. The CSV ranks trials by distance from their own class center, records
the chosen `selected_frame_index` and `stability_score`, and marks whether their
KMeans cluster majority class disagrees with the recorded label. Samples near
the top of the report, especially with `cluster_mismatch=1`, are good
candidates for manual inspection or exclusion before retraining.

Train/validation/test split:

- Each CSV file is treated as one grasp trial. By default, the model first
  cleans the trial by selecting its most stable adjacent-frame pair, then uses
  the first frame of that pair as the feature vector.
- With `data.split_by_trial: true`, `data.loader.load_tactile_trial_table`
  passes each CSV path as a group id.
- `data.splits.split_train_val_indices` shuffles the unique trial ids from
  `data.tactile_root` with `data.random_seed`, then assigns about `val_ratio`
  to validation and the rest to train.
- The default config uses `val_ratio: 0.15`, `random_seed: 42`, so about 85%
  of `Workspace/C_model_training/tactile` trials train and 15% validate.
- `Workspace/C_model_training/tactile_test` is never mixed into the random
  split. It is loaded as the final test set after the best validation checkpoint
  is selected.
- Standardization is fitted only on the training trials, then applied to train,
  validation, and the independent test set.

Thermal liquid-level ablation:

```text
Workspace/C_model_training/thermal/data/merged/*.csv
Workspace/C_model_training/thermal/splits/{train,val,test}.csv
```

Each CSV file is treated as one thermal grasp trial. The split CSVs select the
fixed train/validation/test files; `val.csv` is intentionally copied from
`test.csv`, matching the tactile split convention.

```text
ContainerModel,Ambient_C,Liquid_C,LevelClass,Volume,SourceFile,Index,
base tactile cells..., idw4 tactile interpolation fields...,
AD1_Temperature_C,...,AD8_Temperature_C
```

Run the seven thermal branches:

```powershell
cd "C:\Users\FangYuxuan\Saved Games\Upper_VET6USB\Grasp2Measure"

C:\Users\FangYuxuan\anaconda3\envs\vet6usb_pyqt\python.exe thermal_train.py --config configs/thermal.yaml
```

Outputs:

```text
runs/thermal_ablation/b0_physics/
runs/thermal_ablation/b1_temp_only/
runs/thermal_ablation/b2_weighted_temp_only/
runs/thermal_ablation/b3_no_container_one_hot/
runs/thermal_ablation/b4_no_tau_hat/
runs/thermal_ablation/b5_full_fusion/
runs/thermal_ablation/b6_direct_pressure16_temp/
runs/thermal_ablation/summary_metrics.json
```

Default thermal parameters in `configs/thermal.yaml`:

- Data root: `../Workspace/C_model_training/thermal/data/merged`.
- Split root: `../Workspace/C_model_training/thermal/splits`; current split is 162 train, 28 validation, 28 test, with validation and test using the same file list.
- Steady state: last `8` frames of each CSV are averaged into one trial sample.
- Thermocouple physical order: `AD8, AD3, AD7, AD4, AD1, AD5, AD2, AD6`, corresponding to `L1, L2, L3, L4, R1, R2, R3, R4`.
- B0 sigmoid: `k=0.07715661769546177`, `L=0.8302287969277738`, `x0=55.02710738733247`, `y0=0.08510440076069377`.
- B0 level thresholds: `37.6667 mm`, `49.3333 mm`.
- B0 physics input: sigmoid inversion from the maximum single-channel normalized `DeltaT`; level thresholds are `37.6667 mm`, `49.3333 mm`.
- B1 input: 8-D raw `DeltaT = T_steady - Ambient_C`; hidden width `16`; 227 trainable parameters.
- B2 input: 8-D `alpha * DeltaT`; hidden width `32`.
- B3 input: 17-D `[alpha * DeltaT, alpha, tau_hat]`; hidden width `32`.
- B4 input: 20-D `[alpha * DeltaT, alpha, container_one_hot]`; hidden width `32`.
- B5 input: 21-D `[alpha * DeltaT, alpha, container_one_hot, tau_hat]`; hidden width `32`.
- B6 input: 24-D `[pressure16, raw DeltaT]`; hidden width `32`.
- Tactile neighborhood: each thermocouple uses base tactile cells within `R = 8 mm` from `configs/gripper_sensor_coordinates.csv`.
- Contact mask: `abs(P) > 7 * sigma0`; `sigma0` is robustly estimated from the thermal CSV trial summaries with `sigma_floor: 1.0`.
- Training: `epochs: 200`, `batch_size: 16`, `learning_rate: 0.001`, `weight_decay: 0.0001`.


Current run on the 63 CSV trials:

```text
B0 physics              test_acc=0.1111  macro_f1=0.1111
B1 temp only            test_acc=1.0000  test_loss=0.1694  macro_f1=1.0000
B2 tactile-temp fusion  test_acc=1.0000  test_loss=0.0382  macro_f1=1.0000
```
