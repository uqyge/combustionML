# Compare Inference speed on CPU, GPU and GPU tensorrt

# 1. Cpu and GPU inference performance are compared with Keras model predict( with tensorFlow backend).

- test sample are a array of size (40,000, 2)
- CPU inference typical inference time is 0.1885145902633667 s

- CPU inference time are average over 10 runs. A optimal batch size of 1024\*8 is used. The average
