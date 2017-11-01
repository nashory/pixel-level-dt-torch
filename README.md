# pixel-level-dt-torch
Torch implementation of "[Pixel-Level Domain Transfer](https://arxiv.org/pdf/1603.07442)", bug-fixed version of [repo](https://github.com/fxia22/PixelDTGAN)

![image](https://puu.sh/y8eZp/53c12325b2.png)

## Note (IMPORTANT):
+ I found too many bugs in the [original repo](https://github.com/fxia22/PixelDTGAN), and the code was not running.
+ Finally, I decided to release __bug-fixed__ version of "[Pixel-Level Domain Transfer](https://arxiv.org/pdf/1603.07442)" with enhanced features.
+ The majority of the codes were brought from [original repo](https://github.com/fxia22/PixelDTGAN).

## What is different from original repo?
+ Most importantly, BUG FIXED.
+ The input sizes of both 64 x 64 to 128 x 128 are supported.
+ Network structure was slightly modified acccording to input size.
+ A grid form of generated images are saved in png format.
+ Torch logger is added.

## Prerequisites
+ [Torch7](http://torch.ch/docs/getting-started.html#_)
+ [display](https://github.com/szym/display)

## Dataset preparation
+ (step 1) Download LOOKBOOK dataset: [Here](https://drive.google.com/file/d/0By_p0y157GxQU1dCRUU4SFNqaTQ/view?usp=sharing)
+ (step 2) Place 'lookbook.tar' zip file at root dir.
+ (step 3) run `sh setup.sh`
+ (step 4) preprocess dataset using `prepare_data.ipynb` (you may need to use jupyter notebook)


## How to run?
+ Start training:
~~~
(change training options in opts.lua file beforehand)
python run.py
python run.py & (if want to run in background)
~~~
+ Run server for visualization:
~~~
(change server_ip and server_port options in opts.lua file beforehand)
th server.lua
th server.lua & (if want to run in background)
~~~

## Visualization (display)  
You can see the generated images and loss graphs using web browser.  
`https://<server_ip>:<port>`


## Experimental Results
Result so far (it is still being trained at this moment.)

|training|Final|  
|---|---|  
| condition: 128x128, 0.8 epoch | condition: 128x128, 0.8 epoch |
|<img src="https://github.com/nashory/gif/blob/master/_gans/pixel-level-dt.gif?raw=true" width="400" height="400">|<img src="https://puu.sh/yc5qD/95a1553108.jpg" width="400" height="400">|


## [!!] Trouble-shooting Multi-GPU Memory Allocation issue in torch
It seems that torch uses all memories across the GPUs if you use muti gpu(https://github.com/torch/cutorch/issues/180).
To prevent this and use only single GPU, I gave 'UDA_VISIBLE_DEVICES=n' options when running lua script.
If you are running with single GPU, or do not want to disable multi-gpu memory allocation of torch, remove this option in `run.py`

__(e.g.) os.system('CUDA_VISIBLE_DEVICES=0 th ./main.lua') --> os.system('th ./main.lua')__


## Acknowledgement
+ [@fxia22's original repo](https://github.com/fxia22/PixelDTGAN)


## Author
MinchulShin / [@nashory](https://github.com/nashory)  
__Any insane bug reports or questions are welcome. (min.stellastra[at]gmail.com)  :-)__
