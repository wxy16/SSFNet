# SSFNet
**2024.3.27**

FRFB, BFM, and FAC were updated.

SSFNet： Lightweight real-time network for semantic segmentation of UAV remote sensing images

Our results have been submitted to the official dataset under the username wxy07496, and the specific scores are as follows. Due to the progress of other work, the official website at https://codalab.lisn.upsaclay.fr/competitions/7302#results now displays another project of ours. If interested, please check it out at https://github.com/wxy16/MLFMNet.


![图片1](https://github.com/wxy16/SSFNet/assets/128227957/c2fa5f37-51a1-4191-a8bd-f34d9f9b4b3a)



**UAVid**
```
python tools/uavid_patch_split.py \
--input-dir "data/uavid/uavid_train_val" \
--output-img-dir "data/uavid/train_val/images" \
--output-mask-dir "data/uavid/train_val/masks" \
--mode 'train' --split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024
```

```
python tools/uavid_patch_split.py \
--input-dir "data/uavid/uavid_train" \
--output-img-dir "data/uavid/train/images" \
--output-mask-dir "data/uavid/train/masks" \
--mode 'train' --split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024
```

```
python tools/uavid_patch_split.py \
--input-dir "data/uavid/uavid_val" \
--output-img-dir "data/uavid/val/images" \
--output-mask-dir "data/uavid/val/masks" \
--mode 'val' --split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024
```

## Training

"-c" means the path of the config, use different **config** to train different models.

```
python train_supervision.py -c config/uavid/SSFNet.py
```

## Testing

"-c" denotes the path of the config, Use different **config** to test different models. 

"-o" denotes the output path 

"-t" denotes the test time augmentation (TTA), can be [None, 'lr', 'd4'], default is None, 'lr' is flip TTA, 'd4' is multiscale TTA

"--rgb" denotes whether to output masks in RGB format

**UAVid** ([Online Testing](https://codalab.lisn.upsaclay.fr/competitions/7302))
```
python inference_uavid.py \
-i 'data/uavid/uavid_test' \
-c config/uavid/SSFNet.py \
-o results/uavid/ \
-t 'lr' -ph 1152 -pw 1024 -b 2 -d "uavid"


**Training Code Reference **

 ([GeoSeg](https://github.com/WangLibo1995/GeoSeg))
 ([mmsegmentation](https://github.com/open-mmlab/mmsegmentation))

