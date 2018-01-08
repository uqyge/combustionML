# to play with the ANN toolbox

from ANN_realgas_toolbox import ANN_realgas_toolbox

ANN = ANN_realgas_toolbox()

ANN.import_data()
ANN.scale_split_data()
ANN.setSequential(hiddenLayer=8,n_neurons=500)
ANN.fitModel()
ANN.predict()

# set up the Sequential ANN
ANN.setSequential(hiddenLayer=1,n_neurons=100,loss='mse', optimizer='adam',)  # 5 hidden layer, loss='mse' 'mean_squared_logarithmic_error'

#set up the Resnet ANN
#ANN.setResnet()

ANN.fitModel(batch_size=1000, epochs=400)
ANN.predict()
ANN.plotPredict()
ANN.plotLoss()
