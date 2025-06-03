

# Generating samples for covariance to update prototype in few-shot class-incremental learning  (GCov-FSCIL)

## Dataset
We provide the source code on three benchmark datasets, i.e., CIFAR100, CUB200 and miniImageNet. Please follow the guidelines in [CEC](https://github.com/icoz69/CEC-CVPR2021) to prepare them.

The split of ImageNet100/1000 is availabel at [Google Drive](https://drive.google.com/drive/folders/1IBjVEmwmLBdABTaD6cDbrdHMXfHHtFvU?usp=sharing).

## Code Structures
There are four parts in the code.
 - `models`: It contains the backbone network and training protocols for the experiment.
 - `data`: Images and splits for the data sets.
- `dataloader`: Dataloader of different datasets.
 - `checkpoint`: The weights and logs of the experiment.
 - `generator`: Tt is the backbone network and training protocols for experiment.
 
## Training scripts

- Train CIFAR100

  ```
  python train.py -projec 'GCov-FSCIL' -dataset cifar100  -base_mode "ft_cos" -new_mode "avg_cos" -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 600 -schedule Cosine -gpu 0,1,2,3 -temperature 16 -batch_size_base 256   -balance 0.001 -loss_iter 0 -alpha 0.5
  ```
  
- Train CUB200
    ```
  python train.py -project 'GCov-FSCIL' -dataset cub200  -base_mode ft_cos -new_mode avg_cos -gamma 0.25 -lr_base 0.005 -lr_new 0.1 -decay 0.0005  -schedule Milestone -milestones 50 100 150 200 250 300 -gpu 0,1,2,3 -temperature 16 -batch_size_base 256 -dataroot YOURDATAROOT -balance 0.01 -loss_iter 0 -alpha 0.5 -softmax_t 16 -shift_weight 0.1 -epochs_base 400
  ```

- Train miniImageNet
    ```
    python train.py -project 'GCov-FSCIL' -dataset mini_imagenet -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 1000 -schedule Cosine  -gpu 1,2,3,0 -temperature 16 -dataroot YOURDATAROOT -alpha 0.5 -balance 0.01 -loss_iter 150 -eta 0.1 
    ```

Remember to change `YOURDATAROOT` into your own data root, or you will encounter errors.


