
## Usage
### Installation
Create and activate a conda environment:
```
conda create -n vipt python=3.7
conda activate vipt
```
Install the required packages:
```
bash install_vipt.sh
```

### Data Preparation
Put the training datasets in ./data/. It should look like:
```
$<PATH_of_ViPT>
-- data
    -- DepthTrackTraining
        |-- adapter02_indoor
        |-- bag03_indoor
        |-- bag04_indoor
        ...
    -- LasHeR/train/trainingset
        |-- 1boygo
        |-- 1handsth
        ...
    -- VisEvent/train
        |-- 00142_tank_outdoor2
        |-- 00143_tank_outdoor2
        ...
        |-- trainlist.txt
```

### Path Setting
Run the following command to set paths:
```
cd <PATH_of_ViPT>
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
You can also modify paths by these two files:
```
./lib/train/admin/local.py  # paths for training
./lib/test/evaluation/local.py  # paths for testing
```


### Testing
#### For RGB-D benchmarks
[DepthTrack Test set & VOT22_RGBD]\
These two benchmarks are evaluated using [VOT-toolkit](https://github.com/votchallenge/toolkit). \
You need to put the DepthTrack test set to```./Depthtrack_workspace/``` and name it 'sequences'.\
You need to download the corresponding test sequences at```./vot22_RGBD_workspace/```.

```
bash eval_rgbd.sh
```

#### For RGB-T benchmarks
[LasHeR & RGBT234] \
Modify the <DATASET_PATH> and <SAVE_PATH> in```./RGBT_workspace/test_rgbt_mgpus.py```, then run:
```
bash eval_rgbt.sh
```
We refer you to [LasHeR Toolkit](https://github.com/BUGPLEASEOUT/LasHeR) for LasHeR evaluation, 
and refer you to [MPR_MSR_Evaluation](https://sites.google.com/view/ahutracking001/) for RGBT234 evaluation.


#### For RGB-E benchmark
[VisEvent]\
Modify the <DATASET_PATH> and <SAVE_PATH> in```./RGBE_workspace/test_rgbe_mgpus.py```, then run:
```
bash eval_rgbe.sh
```
We refer you to [VisEvent_SOT_Benchmark](https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark) for evaluation.


