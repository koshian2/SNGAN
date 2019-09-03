# SNGAN

# Reimplementation
Experiments mostly based on the paper.

## CIFAR-10 (Standard CNN)
### Fixed conditions
* Generator: Standard CNN 
* Discriminator : Standard CNN
* n_epochs : 321 (approx. 50k G updates)
* n_dis = 5
* Adam parameters : lr=0.0002, beta1=0.0, beta2 = 0.9
* Using training data (10 classes, 50k images)

[Insert figure here]

### Changing conditions
|       Case      |       0       |   1   |       2       |   3   |
|:---------------:|:-------------:|:-----:|:-------------:|:-----:|
|       Loss      | Cross Entropy | Hinge | Cross Entropy | Hinge |
|   Conditional   |     FALSE     | FALSE |      TRUE     |  TRUE |
| Inception Score |     5.844     | 6.077 |     6.094     | 5.821 |

### Incpetion score log
![](https://raw.githubusercontent.com/koshian2/SNGAN/master/graph/cifar.png)


### Sampling and Interpolation
#### Case 0
IS = 5.844

![](https://raw.githubusercontent.com/koshian2/SNGAN/master/sampling_interpolation/cifar_case0.png)

#### Case 1
IS = 6.077

![](https://raw.githubusercontent.com/koshian2/SNGAN/master/sampling_interpolation/cifar_case1.png)

#### Case 2
IS = 6.094 (**Best**)

![](https://raw.githubusercontent.com/koshian2/SNGAN/master/sampling_interpolation/cifar_case2.png)

#### Case 3
IS = 5.821

![](https://raw.githubusercontent.com/koshian2/SNGAN/master/sampling_interpolation/cifar_case3.png)


## CIFAR-10 (ResNet)
### Fixed conditions
* Generator: ResNet (32x32) 
* Discriminator : ResNet (32x32)
* n_epochs : 321 (approx. 50k G updates)
* n_dis = 5
* Adam parameters : lr=0.0002, beta1=**0.5**, beta2 = 0.9
* Using training data (10 classes, 50k images)

Note: beta1=0.0 failed to converge (seems to stuck into saddle points).

[Insert figure here]

### Changing conditions
|       Case      |       0       |   1   |       2       |   3   |
|:---------------:|:-------------:|:-----:|:-------------:|:-----:|
|       Loss      | Cross Entropy | Hinge | Cross Entropy | Hinge |
|   Conditional   |     FALSE     | FALSE |      TRUE     |  TRUE |
| Inception Score |     3.916     | 5.962 |     3.908     | 5.900 |

Hinge loss performs well.

### Incpetion score log
![](https://raw.githubusercontent.com/koshian2/SNGAN/master/graph/cifar_resnet.png)

### Sampling and Interpolation
#### Case 0
IS = 3.916

![](https://raw.githubusercontent.com/koshian2/SNGAN/master/sampling_interpolation/cifar_resnet_case0.png)

#### Case 1
IS = 5.962 (**Best**)

![](https://raw.githubusercontent.com/koshian2/SNGAN/master/sampling_interpolation/cifar_resnet_case1.png)

#### Case 2
IS = 3.908

![](https://raw.githubusercontent.com/koshian2/SNGAN/master/sampling_interpolation/cifar_resnet_case2.png)

#### Case 3
IS = 5.900 

![](https://raw.githubusercontent.com/koshian2/SNGAN/master/sampling_interpolation/cifar_resnet_case3.png)


## STL-10 (Standard)
### Fixed conditions
* Generator: Standard CNN (48x48) 
* Discriminator : Standard CNN (48x48)
* n_epochs : 1301 (approx. 53k G updates)
* n_dis = 5
* Hinge loss
* Adam parameters : lr=0.0002, beta2 = 0.9
* Using training + test data (10 classes, 13k (5k + 8k) images)

[Insert figure here]

### Changing conditions
|     Case    |   0   |   1   |   2   |   3   |
|:-----------:|:-----:|:-----:|:-----:|:-----:|
| Conditional | FALSE | FALSE |  TRUE |  TRUE |
|    Beta1    |  0.5  |   0   |  0.5  |   0   |
|     IS      | 5.932 | 6.058 | 6.157 | 5.813 |

GANs in STL-10 is bit difficult than CIFAR-10 (Too sensitive to hyper parameters.).

### Incpetion score log
![](https://raw.githubusercontent.com/koshian2/SNGAN/master/graph/stl.png)

### Sampling and Interpolation
#### Case 0
IS = 5.932

![](https://raw.githubusercontent.com/koshian2/SNGAN/master/sampling_interpolation/stl_case0.png)

#### Case 1
IS = 6.058

![](https://raw.githubusercontent.com/koshian2/SNGAN/master/sampling_interpolation/stl_case1.png)

#### Case 2
IS = 6.157 (**Best**)

![](https://raw.githubusercontent.com/koshian2/SNGAN/master/sampling_interpolation/stl_case2.png)

#### Case 3
IS = 5.813

![](https://raw.githubusercontent.com/koshian2/SNGAN/master/sampling_interpolation/stl_case3.png)

## STL-10 (Res Net)
### Fixed conditions
* Generator: Res Net (48x48) 
* Discriminator : Res Net (48x48)
* n_epochs : 1301 if n_dis=5, 261 if n_dis=1 (approx. 53k G updates)
* Hinge loss
* Adam parameters : lr=0.0002, beta1=0.5, beta2 = 0.9
* Using training + test data (10 classes, 13k (5k + 8k) images)

[Insert figure here]

### Changing conditions
|     Case     |   0   |   1   |   2   |   3   |   4   |   5   |   6   |   7   |
|:------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|  Conditional | FALSE |  TRUE | FALSE |  TRUE | FALSE |  TRUE | FALSE |  TRUE |
| N dis update |   5   |   5   |   1   |   1   |   5   |   5   |   1   |   1   |
|   D last ch  |  512  |  512  |  512  |  512  |  1024 |  1024 |  1024 |  1024 |
|      IS      | 5.795 | 7.136 | 3.421 | 3.883 | 6.538 | 7.314 | 3.444 | 3.653 |

Trainings in n_dis = 1 are very difficult. Some additional experiments are written in appendix.

### Incpetion score log
![](https://raw.githubusercontent.com/koshian2/SNGAN/master/graph/stl_resnet.png)

### Sampling and Interpolation
#### Case 0
IS = 5.795

![](https://raw.githubusercontent.com/koshian2/SNGAN/master/sampling_interpolation/stl_resnet_case0.png)

#### Case 1
IS = 7.136 (**2nd**)

![](https://raw.githubusercontent.com/koshian2/SNGAN/master/sampling_interpolation/stl_resnet_case1.png)

#### Case 2
IS = 3.421

![](https://raw.githubusercontent.com/koshian2/SNGAN/master/sampling_interpolation/stl_resnet_case2.png)

#### Case 3
IS = 3.883

![](https://raw.githubusercontent.com/koshian2/SNGAN/master/sampling_interpolation/stl_resnet_case3.png)

#### Case 4
IS = 6.538

![](https://raw.githubusercontent.com/koshian2/SNGAN/master/sampling_interpolation/stl_resnet_case4.png)

#### Case 5
IS = 7.314 (**Best**)

![](https://raw.githubusercontent.com/koshian2/SNGAN/master/sampling_interpolation/stl_resnet_case5.png)

#### Case 6
IS = 3.444

![](https://raw.githubusercontent.com/koshian2/SNGAN/master/sampling_interpolation/stl_resnet_case6.png)

#### Case 7
IS = 3.653

![](https://raw.githubusercontent.com/koshian2/SNGAN/master/sampling_interpolation/stl_resnet_case7.png)

# Additional Implementation
Experiments with datasets that were not written in the paper.

## AnimeFace Character Dataset
[http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/](http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/)

14490 images with 176 classes. 
Much less images per class than CIFAR-10 and STL-10.

### Fixed conditions
* Generator: Res Net (96x96) 
* Discriminator : Res Net (96x96)
* n_epochs : 1301 (approx. 53k G updates)
* Hinge loss, n_dis = 5
* Adam parameters : lr=0.0002, beta1=0.5, beta2 = 0.9

The network architecture is based on a 128x128 paper and the resolution is changed to 96x96 (dense : 4x4x1024 -> 3x3x1024).

[Insert figure here]

### Changing conditions
* Case 0 = unconditional
* Case 1 = conditional 

Inception score was not measured. Because the domain is different and pre-trained inception is useless.

Sampling and interpolation uses the final epoch model.

### Case 0 (Un-conditional)

![](https://raw.githubusercontent.com/koshian2/SNGAN/master/sampling_interpolation/anime_case0.png)

### Case 1 (Conditional)

![](https://raw.githubusercontent.com/koshian2/SNGAN/master/sampling_interpolation/anime_case1.png)

## Oxford Flower Dataset
[http://www.robots.ox.ac.uk/~vgg/data/flowers/](http://www.robots.ox.ac.uk/~vgg/data/flowers/)

8189 images with 102 classes. 
Much less images per class than CIFAR-10 and STL-10.

### Changing conditions
* Case 0 = unconditional
* Case 1 = conditional 

Sampling and interpolation uses the final epoch model.

### Case 0 (Un-conditional)

![](https://raw.githubusercontent.com/koshian2/SNGAN/master/sampling_interpolation/flower_case0.png)

### Case 1 (Conditional)

![](https://raw.githubusercontent.com/koshian2/SNGAN/master/sampling_interpolation/flower_case1.png)

It seems that the generation of flowers is still difficult (outlines are too complicated).

# Appendix (More expertiments on STL-10)
The goal is to train with n_dis = 1, but there are many negative results

## STL-10 (Post-act Res Net / Standard CNN)
### Fixed conditions
* Generator: Post-act Res Net (48x48) 
* Discriminator : Standard CNN (48x48)
* Generator leraning rate : 0.0002

For generator, ResNet implementation in the paper was pre-act ResNet (BN / SN-> ReLU-> Conv), but change this to post-act (Conv-> BN / SN-> ReLU). 

For discriminator, simply used Standard CNN to reduce computation.

【Insert code here】

### Changing conditions
|          Case         |    0   |    1   |    2   |    3   |    4   |    5   |    6   |    7   |   8   |   9   |
|:---------------------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:-----:|:-----:|
|         N dis         |    5   |    5   |    1   |    1   |    1   |    1   |    1   |    1   |   1   |   1   |
|     Beta2 in Adam     |   0.9  |   0.9  |   0.9  |   0.9  |  0.999 |  0.999 |  0.999 |  0.999 | 0.999 | 0.999 |
| Leaky relu slope in D |   0.1  |   0.1  |   0.1  |   0.1  |   0.1  |   0.1  |   0.2  |   0.2  |  0.2  |  0.2  |
|    D learning rate    | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.001 | 0.001 |
|      Conditional      |  FALSE |  TRUE  |  FALSE |  TRUE  |  FALSE |  TRUE  |  FALSE |  TRUE  | FALSE |  TRUE |
|    Inception Score    |  6.419 |  5.663 |  1.285 |  2.634 |  2.342 |  2.447 |  2.499 |  2.722 | 2.355 | 2.544 |

* case 8, 9 = inbalanced learning rate, G:0.0002, D=0.001

| Case | Condition              |   IS  |
|------|------------------------|:-----:|
| 0    | uncoditional n_dis = 5 | 6.419 |
| 2    | uncoditional n_dis = 1 | 1.285 |
| 4    | + beta2 = 0.999        | 2.342 |
| 6    | + lrelu slope = 0.2    | 2.499 |
| 8    | + lr_d = 0.001         | 2.355 |

| Case | Condition            |   IS  |
|------|----------------------|:-----:|
| 1    | coditional n_dis = 5 | 5.663 |
| 3    | coditional n_dis = 1 | 2.634 |
| 5    | + beta2 = 0.999      | 2.447 |
| 7    | + lrelu slope = 0.2  | 2.722 |
| 9    | + lr_d = 0.001       | 2.544 |

It's not good at training with n_dis = 1. Other tunings are also subtle.

### Incpetion score log
![](https://raw.githubusercontent.com/koshian2/SNGAN/master/graph/stl_resnet_postact.png)

### Sampling and Interpolation
Only a part is described. The rest, please refer to them in the repository folder.

#### Case 0
IS = 6.419

![](https://raw.githubusercontent.com/koshian2/SNGAN/master/sampling_interpolation/stl_resnet_postact_case0.png)

#### Case 1
IS = 5.663

![](https://raw.githubusercontent.com/koshian2/SNGAN/master/sampling_interpolation/stl_resnet_postact_case1.png)

#### Case 6
IS = 2.499

![](https://raw.githubusercontent.com/koshian2/SNGAN/master/sampling_interpolation/stl_resnet_postact_case6.png)

#### Case 7
IS = 2.722

![](https://raw.githubusercontent.com/koshian2/SNGAN/master/sampling_interpolation/stl_resnet_postact_case7.png)

### Why n_dis=1 don't work ?

