# DCGAN
A PyTorch implementation of DCGAN.

## Related Papers

Mehdi Mirza and Simon Osindero: Conditional Generative Adversarial Nets. 
https://arxiv.org/pdf/1411.1784.pdf

## Training

```shell
% docker build -t dcgan .
% ./train.sh
```

## Checking Training Results

```shell
% ./run_tensorboard.sh
```