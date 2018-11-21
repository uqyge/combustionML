# Compare Inference speed on CPU, GPU and GPU tensorrt

# 1. Cpu and GPU inference performance are compared with Keras' model.predict( with tensorFlow backend).

- test sample are a array of size **(40,000, 2)**
- CPU inference typical inference time is **188.5145902633667 ms**

- GPU/keras typical inference time is **15.262913703918456 ms**

```python
    there are  10 batches
    Batch inference time is  0.00194091796875
    sequential inference time is  0.0194091796875
    Gpu inference time is  0.018292696475982667
```

- Tensorrt optimazed GPU inference time is around **0.689 ms**.

```c
Average over 10 runs is 0.664858 ms (host walltime is 0.811777 ms, 99% percentile time is 0.676288).
Average over 10 runs is 0.67623 ms (host walltime is 0.821822 ms, 99% percentile time is 0.706816).
Average over 10 runs is 0.69615 ms (host walltime is 0.809204 ms, 99% percentile time is 0.733824).
Average over 10 runs is 0.69849 ms (host walltime is 0.863253 ms, 99% percentile time is 0.791904).
Average over 10 runs is 0.664166 ms (host walltime is 0.802674 ms, 99% percentile time is 0.677568).
Average over 10 runs is 0.710509 ms (host walltime is 0.859756 ms, 99% percentile time is 0.736896).
Average over 10 runs is 0.720682 ms (host walltime is 0.84437 ms, 99% percentile time is 0.733792).
Average over 10 runs is 0.693878 ms (host walltime is 0.845249 ms, 99% percentile time is 0.73312).
Average over 10 runs is 0.708749 ms (host walltime is 0.835532 ms, 99% percentile time is 0.73504).
Average over 10 runs is 0.713789 ms (host walltime is 0.841595 ms, 99% percentil
```

- The inference speed scales are **1/12.3/27.3**( CPU/GPU/tensorrt).

- CPU inference are parallelised over 4 cores. A optimal batch size of 1024\*8 is used.

- The results shows there are room for stream concurency optimaztion.( The GPU inferenence time is short that accumulated batch inference time.)

- Tensorrt optimazed GPU inference time are predicted based on a given batchsize 4096 with trtexec.

```c
trtexec --uff=mayer.uff --uffInput=input_1,1,2,1, --output=dense_2/BiasAdd  --batch=4096
```
