# 3D virtual H&E staining
![Overview of virtual H&E staining](images/fig1.jpg)

This repository contains codes and datasets necessary for testing the 3D virtual H&E staining of label-free holotomography images of tissue slides.

## Download datasets and model before testing
Pretrained models and example datasets : [link](https://doi.org/10.57760/sciencedb.24217) 

## Installation
```shell
conda env create -f environment.yml
```

```shell
conda activate virtualstaining
``` 

## Set the directories

```shell

dataset                
 ├──  test.yaml
 ├──  test     
 |      ├── 001.h5     
 |      ├── ...
 |      └── xxx.h5
 └──

model-gastric                
 ├──  ckpt     
 |      └── model_gastric.pth
 └──

codes
 └── codes downloaded from this repository

```

## Use appropriate path for the following parameters
In the main.py, 

``` shell

--data_dir = 'xxx/dataset'
--ckpt_dir = 'xxx/model-gastric/ckpt'
--result_dir = 'where you want to save the results'

```


## Run the code for training

```shell

python main.py --network scnas --mode train --batch_size 1

```

## Run the code for testing

```shell

python main.py --network scnas --mode test --batch_size 1

```
The training and testing were all performed with the options of --batch_size 1 and --network scnas


## License
This project is open-sourced under the MIT license.


