# to play with the ANN toolbox

import matplotlib.pyplot as plt
import datetime

from ANN_realgas_toolbox import ANN_realgas_toolbox

ANN = ANN_realgas_toolbox()

ANN.import_data()
#ANN.scale_split_data()
#ANN.scale_split_data(targets = ['Cp'])
#ANN.setSequential(hiddenLayer=1,n_neurons=100)
#ANN.fitModel()
#ANN.prediction()
ANN.scale_split_data()
start = datetime.datetime.now()
ANN.gridSearchSequential(epochs = [400], batch= [500],loss_func = ['mse','mae','mape','msle','squared_hinge','hinge','categorical_hinge','logcosh','kullback_leibler_divergence','poisson','cosine_proximity'])
end = datetime.datetime.now()

ANN.plotAccuracy(target='Cp')

#set up the Resnet ANN
# ANN.setResnet()
# ANN.fitModel(batch_size=1000, epochs=400)
# ANN.predict()
# ANN.plotPredict()
# ANN.plotLoss()
