# CardinalBlue Interview Take-Home Question 2

I was asked to reproduce the results of this [paper](https://arxiv.org/pdf/1711.00489v1.pdf).

However, I don't have to use the same neural network architecture as the paper.

The goal is to observe the trade-off between batch size and the learning rate.

So I'll show few plots and friefly summarize what I learned from the experiments.

If you want to see the detail of my code for the results, please refer to `Experiments.ipynb`.

## Training Environment

```
# nvidia-smi
Wed Nov 22 14:25:27 2017       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 370.28                 Driver Version: 370.28                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 980 Ti  Off  | 0000:02:00.0     Off |                  N/A |
|  0%   54C    P8    33W / 275W |      1MiB /  6076MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                         
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```

```
# tensorflow.__version__
1.3.0
```

## Setup

Since I was allowed to use a simpler architecture, my experiment setting is not the same as settings in the paper.

I want to show that you can either training with decreasing learning rate or increasing batch size to get similer results such as training loss or accuracy.

As a result, there are two different setups in my experiment:
1. Learning Rate Decay (`lrd` for short)
    - Fixed batch size: 128
    - Decreasing learning rate:
    - initial value: 0.1
    - decay rate: 0.9
    - learning rate updated after every epoch
2. Increasing Batch Size (`ibs` for short)
    - Increasing batch size:
    - initial value: 128
    - maximum batch size: 4526
        - `(# of training data)/10`
    - growth rate: 1.037
        - It will reach maximum batch size at approximately 100th epoch

The goal is to obtain similer results (training loss, accuracy...etc) with these two different setups.

The network is trained with simple **SGD** in both cases.

## Network Architecture

<img alt=simple-cnn src=images/simple_cnn.png width=600px height=600px />

- `conv1`:
    - kernel size: 3x3
    - \# of kernels: 32
    - stride: 2
    - padding: 'VALID'
    - relu activation function
- `conv2`:
    - kernel size: 3x3
    - \# of kernels: 32
    - stride: 2
    - padding: 'VALID'
    - relu activation function
- `pool1`:
    - max pooling
    - kernel size: 2x2
    - stride: 1
    - padding: 'VALID'
- `conv3_1_by_1`:
    - kernel size: 1x1
    - \# of kernels: 16
    - stride: 1
    - padding: 'SAME'
    - relu activation function
- `fc1`:
    - hidden layer size: 512
    - relu activation function
- `fc2`:
    - the output layer
    - softmax

Network architecture capacity is checked by overfitting over a small fraction of traing data (1000 out of 45235)

```
...
step: 3000
loss: 0.0980233
  accuracy: 0.997
step: 3500
loss: 0.0314949
  accuracy: 1.0
step: 4000
loss: 0.0152432
  accuracy: 1.0    # overfitting!
```

## Result Plots

<img alt=final-results src=images/final_results.png width=900px width=600px />

- Left Plot (Traininig Loss)
    1. Though the convergent rate is not comparable in terms of iterations in both cases, we can still say that the loss converge to low value with much fewer parameter updates for `ibs`
    2. In fact, the final training loss is even lower in `ibs` than the loss in `lrd`
- Right Plot (Test Accuracy)
    1. Testing accuracy grows faster in the case of `ibs`
    2. `ibs` converges to higher testing accuracy than `lrd` in the end

You can find the training log files in the `log` directory.
- `ibs_graph/bs_{init batch size}_{growth rate}_{learn rate}/train_log.txt`
- `lrd_graph/lr_{init learn rate}_{decay rate}/train_log.txt`

## Summary

1. According to the results above, I think I do reproduce some results in the paper
2. However, I have to admit that values in the setting, such as initial learnin rate, decay rate,...etc, are selected with care by me.
    - In the `Experiments.ipynb`, you can play with `lrd_train` and `ibs_train` functions to see how it works with different initial values.
    - Sometime, the model just explode (with `NaN` in the filters). Maybe trying different activation function can help in this case but I don't have time to do that.
3. In practice, I think I can come up with following guilds:
    - If you have a GPU with big memory available, maybe it is a good idea to try increasing batch size 
    - If there is no room for large batch on your GPU, just go for learing rate decay

## TODO

- Hybrid training setup

**If you think I've done wrong with anything, PRs are welcomed. :)**