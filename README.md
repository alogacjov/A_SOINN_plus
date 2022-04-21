# A-SOINN+
The associative SOINN+ (A-SOINN+) experiments and the v-NICO-World-LL dataset for continual learning proposed in [Learning Then, Learning Now, and Every Second in Between: Lifelong Learning With a Simulated Humanoid Robot](https://doi.org/10.3389/fnbot.2021.669534).

## v-NICO-World-LL

### Original Dataset
The original [v-NICO-World-LL](https://www2.informatik.uni-hamburg.de/wtm/datasets/20220419_v_NICO_World_LL/) dataset can be downloaded with the command:
```bash
wget -r --no-parent -nH --cut-dirs=2 https://www2.informatik.uni-hamburg.de/wtm/datasets/20220419_v_NICO_World_LL/
```
Note that the files are compressed and have to be decompressed with the command:
```bash
tar -zxvf s<i>.tgz
# Example: tar -zxvf 20220419_v_NICO_World_LL/s0.tgz
```

### Feature Vectors
You don't have to download the dataset to start the experiments. For the experiments, use [v-NICO-World-LL-feature-vectors](https://github.com/alogacjov/A_SOINN_plus/tree/main/data/v_NICO_World_LL_feature_vectors). The folder contains feature vectors for each image of the original v-NICO-World-LL dataset. These features are created using a pre-trained VGG16 model as described in our [paper](https://doi.org/10.3389/fnbot.2021.669534).


## Experiments
### Requirements
- Python 3.8.10+
```bash
pip install -r requirements.txt
```

### Usage
Both the A-SOINN+ and the [Growing-Dual Memory (GDM)](https://doi.org/10.3389/fnbot.2018.00078), proposed by German I. Parisi et al., can be trained with the following command:
```bash
./run_training.sh -c <path/to/model/config.yml> -d <path/to/dataset>
# Example A-SOINN+: ./run_training.sh -c src/configs/asoinn_plus/config.yml -d data/v_NICO_World_LL_feature_vectors
# Example GDM: ./run_training.sh -c src/configs/gdm/config.yml -d data/v_NICO_World_LL_feature_vectors
```

The config files of the [GDM](https://github.com/alogacjov/A_SOINN_plus/blob/main/src/configs/gdm/config.yml) and [A-SOINN+](https://github.com/alogacjov/A_SOINN_plus/blob/main/src/configs/asoinn_plus/config.yml) approaches contain hyperparameters that can be adjusted. The default values are the same as in our paper.

### Note
The [GDM reimplementation](https://github.com/alogacjov/A_SOINN_plus/blob/main/src/models/gdm.py) is based on the original GDM implementation of German I. Parisi et al. https://github.com/giparisi/GDM, proposed in:

G. I. Parisi, J. Tani, C. Weber, and S. Wermter, “Lifelong Learning of Spatiotemporal Representations With Dual-Memory Recurrent Self-Organization,” Front. Neurorobot., vol. 12, 2018, [doi: 10.3389/fnbot.2018.00078](https://doi.org/10.3389/fnbot.2018.00078).


## License
The v-NICO-World-LL dataset and the A-SOINN+ approach are distributed under the Creative Commons CC BY-NC-ND 4.0 license. If you use them, you agree (i) to use them for research purposes only, and (ii) to cite the following reference in any works that make any use of the dataset or the approach.

## Citation
A. Logacjov, M. Kerzel, and S. Wermter, “Learning Then, Learning Now, and Every Second in Between: Lifelong Learning With a Simulated Humanoid Robot,” Frontiers in Neurorobotics, vol. 15, p. 78, 2021, [doi: 10.3389/fnbot.2021.669534](https://doi.org/10.3389/fnbot.2021.669534).
