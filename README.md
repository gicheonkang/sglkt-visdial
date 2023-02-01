SGLKT-VisDial
====================================

Pytorch Implementation for the paper:

**[Reasoning Visual Dialog with Sparse Graph Learning and Knowledge Transfer][1]** <br>
[Gi-Cheon Kang](https://gicheonkang.com), Junseok Park, [Hwaran Lee](https://hwaranlee.github.io), [Byoung-Tak Zhang](https://bi.snu.ac.kr/~btzhang/)<sup>\*</sup>, and [Jin-Hwa Kim](http://wityworks.com)<sup>\*</sup> (\* corresponding authors) <br>
In EMNLP 2021 Findings

<!--![Overview of Sparse Graph Learning](sgl_overview.png)-->
<img src="sgl_overview.png" width="90%" align="middle">

Setup and Dependencies
----------------------
This code is implemented using PyTorch v1.7, and provides out of the box support with CUDA 11 and CuDNN 8. Anaconda/Miniconda is the recommended to set up this codebase: <br>

1. Install Anaconda or Miniconda distribution based on Python3+ from their [downloads' site][2].
2. Clone this repository and create an environment:

```shell
git clone https://www.github.com/gicheonkang/sglkt-visdial
conda create -n sglkt python=3.8

# activate the environment and install all dependencies
conda activate sglkt
cd sglkt-visdial/
pip install -r requirements.txt

# install this codebase as a package in development version
python setup.py develop
```

Download Data
----------------------
1. We used the Faster-RCNN pre-trained with Visual Genome as image features. Download the image features below, and put each feature under `$PROJECT_ROOT/data/{SPLIT_NAME}_feature` directory. We need `image_id` to RCNN bounding box index file (`{SPLIT_NAME}_imgid2idx.pkl`) because the number of bounding box per image is not fixed (ranging from 10 to 100).

  * [`train_btmup_f.hdf5`][3]: Bottom-up features of 10 to 100 proposals from images of `train` split (32GB). 
  * [`val_btmup_f.hdf5`][4]: Bottom-up features of 10 to 100 proposals from images of `validation` split (0.5GB).
  * [`test_btmup_f.hdf5`][5]: Bottom-up features of 10 to 100 proposals from images of `test` split (2GB).

2. Download the pre-trained, pre-processed word vectors from [here][6] (`glove840b_init_300d.npy`), and keep them under `$PROJECT_ROOT/data/` directory. You can manually extract the vectors by executing `data/init_glove.py`.

3. Download visual dialog dataset from [here][7] (`visdial_1.0_train.json`, `visdial_1.0_val.json`, `visdial_1.0_test.json`, and `visdial_1.0_val_dense_annotations.json`) under `$PROJECT_ROOT/data/` directory.     

4. Download the additional data for Sparse Graph Learning and Knowledge Transfer under `$PROJECT_ROOT/data/` directory.

  * [`visdial_1.0_train_coref_structure.json`][8]: structural supervision for `train` split.
  * [`visdial_1.0_val_coref_structure.json`][9]: structural supervision for `val` split.
  * [`visdial_1.0_test_coref_structure.json`][10]: structural supervision for `test` split.
  * [`visdial_1.0_train_dense_labels.json`][11]: pseudo labels for knowledge transfer.
  * [`visdial_1.0_word_counts_train.json`][12]: word counts for `train` split.   


Training
--------

Train the model provided in this repository as:

```shell
python train.py --gpu-ids 0 1 # provide more ids for multi-GPU execution other args...
```

### Saving model checkpoints

This script will save model checkpoints at every epoch as per path specified by `--save-dirpath`. Default path is `$PROJECT_ROOT/checkpoints`.

Evaluation
----------

Evaluation of a trained model checkpoint can be done as follows:

```shell
python evaluate.py --load-pthpath /path/to/checkpoint.pth --split val --gpu-ids 0 1
```
Validation scores can be checked in offline setting. But if you want to check the `test split` score, you have to submit a json file to [EvalAI online evaluation server][13]. You can make json format with `--save_ranks True` option.

Pre-trained model & Results
--------
We provide the pre-trained models for [SGL+KT][14] and [SGL][15]. <br>
To reproduce the results reported in the paper, please run the command below.
```shell
python evaluate.py --load-pthpath SGL+KT.pth --split test --gpu-ids 0 1 --save-ranks True
```

Performance on `v1.0 test-std` (trained on `v1.0` train):

  Model  |  Overall | NDCG   |  MRR   |  R@1  | R@5  |  R@10   |  Mean  |
 ------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
  SGL+KT | 65.31 | 72.60 | 58.01 | 46.20 | 71.01 | 83.20 | 5.85

Citation
--------
If you use this code in your published research, please consider citing:
```text
@article{kang2021reasoning,
  title={Reasoning Visual Dialog with Sparse Graph Learning and Knowledge Transfer},
  author={Kang, Gi-Cheon and Park, Junseok and Lee, Hwaran and Zhang, Byoung-Tak and Kim, Jin-Hwa},
  journal={arXiv preprint arXiv:2004.06698},
  year={2021}
}
```

License
--------
MIT License

Acknowledgements
--------
We use [Visual Dialog Challenge Starter Code][16] and [MCAN-VQA][17] as reference code.   

[1]: https://arxiv.org/abs/2004.06698
[2]: https://conda.io/docs/user-guide/install/download.html
[3]: https://www.dropbox.com/s/zxy37v9uloe18qi/train_btmup_f.hdf5?dl=0
[4]: https://www.dropbox.com/s/c33xjfddz45ff6r/val_btmup_f.hdf5?dl=0
[5]: https://www.dropbox.com/s/e7d3mpwe2it2z7y/test_btmup_f.hdf5?dl=0
[6]: https://www.dropbox.com/s/gicspjyysxdlkod/glove840b_init_300d.npy?dl=0
[7]: https://visualdialog.org/data
[8]: https://www.dropbox.com/s/dy3i8ma10ttuqxb/visdial_1.0_train_coref_structure.json?dl=0
[9]: https://www.dropbox.com/s/r1e6j3o4l6j6mvg/visdial_1.0_val_coref_structure.json?dl=0
[10]: https://www.dropbox.com/s/9n2fmnwnwbsxt4y/visdial_1.0_test_coref_structure.json?dl=0
[11]: https://www.dropbox.com/s/fmyo23br7q94uxf/visdial_1.0_train_dense_processed.json?dl=0
[12]: https://www.dropbox.com/s/746cmik1qw2yavx/visdial_1.0_word_counts_train.json?dl=0
[13]: https://eval.ai/web/challenges/challenge-page/518/overview
[14]: https://www.dropbox.com/s/beq10ad1a25tng7/SGL%2BKT.pth?dl=0
[15]: https://www.dropbox.com/s/e9ktzezfl99otqv/SGL.pth?dl=0
[16]: https://www.github.com/batra-mlp-lab/visdial-challenge-starter-pytorch
[17]: https://github.com/MILVLG/mcan-vqa
