# to play with the ANN toolbox

import matplotlib.pyplot as plt


from ANN_realgas_toolbox import ANN_realgas_toolbox

ANN = ANN_realgas_toolbox()

ANN.import_data()
ANN.scale_split_data()
ANN.setSequential(hiddenLayer=5,n_neurons=600)
ANN.fitModel()
ANN.prediction()


ANN.plotAccuracy(target='Cp')
ANN.plotAccuracy(target='rho')


