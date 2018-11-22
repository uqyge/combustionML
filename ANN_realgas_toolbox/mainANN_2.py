# to play with the ANN toolbox

import matplotlib.pyplot as plt
import datetime

from ANN_realgas_toolbox import ANN_realgas_toolbox

ANN = ANN_realgas_toolbox()

ANN.import_data()
ANN.scale_split_data()
#ANN.scale_split_data(targets = ['Cp'])
#ANN.setSequential(hiddenLayer=1,n_neurons=100)
#ANN.fitModel()
#ANN.prediction()
ANN.scale_split_data()
start = datetime.datetime.now()
ANN.gridSearchResNet()
end = datetime.datetime.now()

ANN.plotAccuracy(target='Cp')

