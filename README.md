# FL4WS · Federated Wireless Sensing Benchmark

FL4WS prepares Wi-Fi sensing datasets for federated learning. It provides a common Python interface for loading samples, PyTorch classification baselines, and a command-line interface for generating reproducible client partitions, training and test arrays, and distribution statistics.

## Datasets

All six adapters support IID and label-based Dirichlet partitioning. The **Natural grouping** column lists the additional strategies supported by each release's metadata. Shapes describe one exported sample.

| Dataset / CLI ID | Task and input | Samples | Classes | Sample shape | Natural grouping | Source |
| --- | --- | ---: | ---: | --- | --- | --- |
| UT-HAR / `ut-har` | Human activity recognition with SenseFi CSI windows | 4,973 | 7 | `(1, 250, 90)` | — | [SenseFi data](https://data.mendeley.com/datasets/dzvgyxkx2f/1) |
| NTU-Fi HAR / `ntu-fi-har` | Human activity recognition with SenseFi CSI amplitude | 1,200 | 6 | `(3, 114, 500)` | — | [SenseFi data](https://data.mendeley.com/datasets/dzvgyxkx2f/1) |
| WiAR / `wiar` | Human activity recognition with Intel 5300 recordings from participants A–C | 1,440 | 16 | `(3, 30, 128)` | Cross-Subject (3 subjects) | [WiAR release](https://github.com/linteresa/WiAR) |
| ARIL / `aril` | Arm gesture recognition with CSI amplitude and position annotations | 1,394 | 6 | `(52, 192)` | — | [ARIL release](https://github.com/geekfeiw/ARIL) · [Data mirror](https://www.kaggle.com/datasets/hylanj/wifi-csi-dataset-aril) |
| Widar 3.0 / `widar3` | Hand gesture recognition with SenseFi body-coordinate velocity profiles (BVP) | 43,652 | 22 | `(22, 20, 20)` | Cross-Subject (17 subjects) | [SenseFi data](https://data.mendeley.com/datasets/dzvgyxkx2f/1) · [Widar project](https://tns.thss.tsinghua.edu.cn/widar3.0/) |
| MM-Fi / `mm-fi` | Human activity recognition using the Wi-Fi CSI modality | 1,080 | 27 | `(32, 3, 114, 10)` | Cross-Subject (40 subjects), Cross-Environment (4 scenes) | [MM-Fi release](https://github.com/ybhbingo/MMFi_dataset#download) |

This workspace has local inputs for UT-HAR, NTU-Fi HAR, WiAR, ARIL, and Widar 3.0. The MM-Fi adapter is included and can be run with an extracted copy of the CSI release. Its table counts describe the full release.

### Data layout and preprocessing

Place the source files under `data/` or point `--root` at the appropriate dataset directory.

```text
data/
├── UT_HAR/
│   ├── data/X_{train,val,test}.csv
│   └── label/y_{train,val,test}.csv
├── NTU-Fi_HAR/
│   ├── train_amp/<activity>/*.mat
│   └── test_amp/<activity>/*.mat
├── WiAR/
│   └── volunteer_{a,b,c}_all_data/csi_a<activity>_<repeat>.dat
├── ARIL/
│   └── {train,test}_data_split_amp.mat
├── Widardata/
│   └── Widardata.zip
└── MMFi_Dataset/
    └── E01/S01/A01/wifi-csi/frame001.mat
```

- **UT-HAR** stores NumPy arrays under `.csv` filenames. **NTU-Fi HAR** reads `CSIamp` and keeps every fourth time step. [SenseFi loader](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark/blob/main/dataset.py)
- **WiAR** uses one sample per recording. It removes byte-identical recordings before partitioning and selects 128 packet amplitudes evenly across each recording.
- **ARIL** uses activity as the classification label and records the 16 measurement positions as `location_id` in the sample manifest.
- **Widar 3.0** can read `Widardata.zip` directly. With an extracted release, set `--root` to the directory containing `train/` and `test/`. The file name supplies the subject ID.
- **MM-Fi** selects 32 nonempty CSI frames evenly across each activity recording and uses the amplitude array. The source layout supplies subject, environment, and activity IDs. Each participant was recorded in one scene, so environment groups also differ in participant membership. [MM-Fi collection and loader](https://github.com/ybhbingo/MMFi_dataset)

Each adapter returns a `float32` feature array and a zero-based integer label. Use `--normalization zscore` to standardize each sample.

## Quick start

Install with Python 3.9 or later, then run one of the dataset generators.

```bash
python -m pip install -e .

fl4ws --dataset wiar --root data/WiAR \
  --strategy cross-subject --seed 42 --output outputs/wiar-subject

fl4ws --dataset aril --root data/ARIL \
  --strategy dirichlet --num-clients 10 --alpha 0.3 --output outputs/aril-dirichlet
```

The same commands work with `python -m fl4ws`. Each dataset also has a standalone `dataset/generate_*.py` entry point that takes the same options with the dataset already selected. Common loading, partitioning, and export code lives in `dataset/utils.py`.

Every entry point reads `dataset/config.yaml` by default. Use `--config` to select another YAML file. Command-line values override YAML values, and relative paths are resolved from the working directory.

### Partitioning and evaluation

- **IID** shuffles samples within each class and distributes them evenly across clients. **Dirichlet** samples per-class client proportions using `--alpha`. Both strategies require `--num-clients` and use a balanced assignment for the test pool.
- **Cross-Subject** assigns one client per subject. **Cross-Environment** assigns one client per scene. Each client receives the training and test samples from its own group.
- UT-HAR preserves the published training pool and combines its validation and test splits for evaluation. NTU-Fi HAR, ARIL, and Widar 3.0 preserve the released training and test pools. WiAR and MM-Fi hold out whole recordings within each subject using the configured seed and `--train-ratio`.

## Classification models

Install the optional PyTorch dependency with `python -m pip install -e '.[models]'`. Dataset loading, partitioning, and export do not require PyTorch.

| Dataset IDs | Default baseline | Implementation |
| --- | --- | --- |
| `ut-har`, `ntu-fi-har`, `wiar` | 2D CNN over each CSI window | `models/cnn2d.py` |
| `aril` | 1D temporal CNN with subcarriers as input channels | `models/cnn1d.py` |
| `widar3`, `mm-fi` | Shared 2D frame encoder followed by a GRU | `models/cnn2d_gru.py` |

`build_model` accepts the same dataset ID as `load_dataset`. It takes a `float32` tensor with shape `(batch, *sample_shape)` in the adapter's original axis order and returns raw logits of shape `(batch, num_classes)`. Use these logits directly with cross-entropy and zero-based integer labels.

```python
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

from fl4ws import build_model, load_dataset

dataset = load_dataset("aril", "data/ARIL", normalization="zscore")
model = build_model(dataset.name, num_classes=len(dataset.classes))
x, y = next(iter(DataLoader(dataset, batch_size=8)))
loss = cross_entropy(model(x), y)
loss.backward()
```

This example demonstrates a forward and backward pass; for federated training, use each client's exported `train` and `test` arrays separately. The default class count is for the full release. For a subset, pass `len(dataset.classes)` or the length of `partition.json`'s `classes` list so all clients use the same output dimension, even when individual clients lack some classes.

## Outputs

Each run writes the following files to a new output directory.

- `train/<client_id>.npz` and `test/<client_id>.npz` containing feature arrays `x` and labels `y` under the `data` key.
- `partition.json` containing the label map, preprocessing settings, sample metadata, and client indices.
- `statistics.json` containing class, subject, and environment counts for each client and the full partition.
- `config.yaml` containing the effective settings. Replay the partition with `--config`, the same source directory, and a new `--output` path.

Use `np.load(path, allow_pickle=True)` and `archive["data"].item()` to read an archive you generated or trust. For Python integrations, import `load_dataset`, `PartitionConfig`, `partition_dataset`, and `write_partition` from `dataset`.
