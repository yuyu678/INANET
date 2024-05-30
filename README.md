Introduction
============
This is a PyToch implementation of INA-DBNet.

We propose a new text detection method fusing the pure ConvNet model and the multi-scale attention mechanism. This design fuses local and global features and achieves the balance of accuracy and speed. Experimental results demonstrate the efficacy of our method. Particularly, on the MSRA-TD500 and Total-Text datasets, INA-DBNet achieves 91.3% and 86.7% F-measure while maintaining real-time inference speed.

![image](https://github.com/yuyu678/INANET/blob/main/model.png)

Installation
============
Requirements:
------------
* Python3
* PyTorch == 1.2
* GCC >= 4.9 (This is important for PyTorch)
* CUDA >= 9.0 (10.1 is recommended)

Datasets
========

The root of the dataset directory can be INANet/datasets/.

Download the converted ground-truth and data list  Baidu Drive (download code: 0drc), Google Drive. The images of each dataset can be obtained from their official website.

Testing
=======

Demo
----
Run the model inference with a single image. Here is an example:

```python
CUDA_VISIBLE_DEVICES=0 python demo.py experiments/seg_detector/totaltext_resnet50_deform_thre_INA.yaml --image_path datasets/total_text/test_images/img10.jpg --resume path-to-model-directory/totaltext_resnet50_INA --polygon --box_thresh 0.7 --visualize
```
Evaluate the performance
------------------------
The following command can re-implement the results in the paper:

```python
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/totaltext_resnet50_deform_thre_INA.yaml --resume path-to-model-directory/totaltext_resnet50_INA --polygon --box_thresh 0.6
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/td500_resnet50_deform_thre_INA.yaml --resume path-to-model-directory/td500_resnet50_INA --box_thresh 0.5
CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/ic15_resnet50_deform_thre_INA.yaml --resume path-to-model-directory/ic15_resnet50_INA --box_thresh 0.6
```
box_thresh can be used to balance the precision and recall, which may be different for different datasets to get a good F-measure. polygon is only used for arbitrary-shape text dataset.

Training
========
Check the paths of data_dir and data_list in the base_*.yaml file.
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py path-to-yaml-file --num_gpus 4
```
You can also try distributed training.
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py path-to-yaml-file --num_gpus 4
```


