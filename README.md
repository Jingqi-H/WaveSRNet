Paper: Wavelet-based Selection-and-Recalibration Network for Parkinson's Disease Screening in OCT Images



## Data Availability

**1. Two public OCT datasets:**

Download from the link provided below:

The SD-OCT dataset is released in the paper titled "Fully automated detection of diabetic macular edema and dry age-related macular degeneration from optical coherence tomography images".  [Available link](http://www.duke.edu/~sf59/Srinivasan_BOE_2014_dataset.htm)

The UCSDdataset is released in the paper titled "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning". [Available link](https://data.mendeley.com/datasets/rscbjbr9sj/2?__hstc=25856994.050094848ba039343e49229a8108ceb1.1715736860812.1715736860812.1715736860812.1&__hssc=25856994.1.1715736860812&__hsfp=1392219453)


**2. The private PD-OCT dataset**

The PD-OCT dataset is confidential due to laboratory policies and confidentiality agreements, which can be accessed by contacting the corresponding author (J. Liu: liuj@sustech.edu.cn) upon reasonable request.



## Train

### 1 k_fold:

```
nohup python main_kf_rebuttal.py --modelname res18 --version 20240428R1 --gpu_ids 0 --num_epochs 100 --patience 10 >./checkpoints/logs/kflod_20240428_res18.log
nohup python main_kf_rebuttal.py --modelname wavesrnet --version 20240510R1.1 --gpu_ids 2 --num_epochs 100 --patience 20 --lr 0.01 --batch_size 64 >./checkpoints/logs/kflod_20240510R1.1_wavesrnet.log
```



### 2 train on train/val/test dataset

1. SD-OCT dataset:

   We need to split data first: `./dataloaders/OCT_AMD_new_csv.py`

```
nohup python main.py --modelname dwan --version 20230802R1 --gpu_ids 2 --batch_size 64 --num_epochs 100 --patience 10 --val_bs 1 --root ./dataloaders --dataset_factory OCT_AMD_R1 --num_classes 3 --public_data >./checkpoints/logs/oh_20230802R1_OCTAMDR1_dwan.log
```

2. USCD dataset:

```
nohup python main.py --modelname res18 --version 20230610R1 --gpu_ids 1 --batch_size 64 --num_epochs 100 --patience 10 --val_bs 2 --root /disk1/imed_hjq/data --dataset_factory USCD --num_classes 4 >./checkpoints/logs/oh_20230610R1_USCD_res18.log

nohup python main.py --modelname wavesrnet --version 20240123R1 --gpu_ids 6 --batch_size 64 --num_epochs 100 --patience 10 --val_bs 2 --root /data1/hjq/data --dataset_factory USCD --num_classes 4 >./checkpoints/logs/oh_20240123R1_USCD_wavesrnet.log
```

3. PD dataset:

```
nohup python main.py --modelname wavesrnet --version 20240123R2 --gpu_ids 6 --batch_size 64 --num_epochs 100 --patience 1 --val_bs 2 --root /data1/hjq/data/University/Parkinsonism/oriR2 --dataset_factory disk12data1_oriR2_split_fileR1 >./checkpoints/logs/oh_20240123R2_oriR2_split_fileR1_wavesrnet.log
```



## Reference

Our implementation builds upon several existing publicly available codes.
- [CLAM](https://github.com/mahmoodlab/CLAM)
- [FRMIL](https://github.com/PhilipChicco/FRMIL/tree/main)
- [DAWN](https://github.com/mxbastidasr/DAWN_WACV2020/tree/5fc336575ad7900173fe08b4b0f32a44492161b3)
