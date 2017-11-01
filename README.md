# pixel-level-dt-torch
Torch implementation of "[Pixel-Level Domain Transfer](https://arxiv.org/pdf/1603.07442)", bug-fixed version of [repo](https://github.com/fxia22/PixelDTGAN)

![image](https://puu.sh/y8eZp/53c12325b2.png)

## Note (IMPORTANT):
+ I found too many bugs in the [original repo](https://github.com/fxia22/PixelDTGAN), and the code was not running.
+ Finally, I decided to release __bug-fixed__ version of "[Pixel-Level Domain Transfer](https://arxiv.org/pdf/1603.07442)" with enhanced features.
+ The majority of the codes were brought from [original repo](https://github.com/fxia22/PixelDTGAN).

## What is different from original repo?
+ Most importantly, BUG FIXED.
+ The input size has been changed from 64 x 64 to 128 x 128.
+ Network structure was slightly modified.

## Prerequisites
+ [Torch7](http://torch.ch/docs/getting-started.html#_)
+ [display](https://github.com/szym/display)


## How to run?
to be updated...


## Experimental Results
Result so far (it is stil being trained at this moment.)

|training|Final|  
|---|---|  
| condition: 128x128, 0.8 epoch | condition: 128x128, 0.8 epoch |
|<img src="https://github.com/nashory/gif/blob/master/_gans/pixel-level-dt.gif?raw=true" width="400" height="400">|<img src="https://puu.sh/yc5qD/95a1553108.jpg" width="400" height="400">|



## Acknowledgement
+ [@fxia22's original repo](https://github.com/fxia22/PixelDTGAN)


## Author
MinchulShin / [@nashory](https://github.com/nashory)
__Any insane bug reports or questions are welcome. (min.stellastra[at]gmail.com)  :-)__
