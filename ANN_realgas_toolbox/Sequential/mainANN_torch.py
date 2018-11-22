# to play with the ANN toolbox

import matplotlib.pyplot as plt
import datetime

from ANN_realgas_toolbox_torch import ANN_realgas_toolbox_torch, createDataset

ANN = ANN_realgas_toolbox_torch()

ANN.import_data()
ANN.scale_split_data()
ANN.setSequential()
#ANN.fitModel()
ANN.fitTorch(epochs = 5000000,learning_rate=1e-3)
#ANN.run(10,0.00001,0.9)
