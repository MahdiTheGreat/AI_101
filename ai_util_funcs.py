def train_test_split_with_indices(X, y, test_size,dtype):
 # usefull when using sklearn.model_selection like the example below

 #  from sklearn.model_selection import GridSearchCV,StratifiedKFold
 #  parameters={
 # 'C': [0.01,0.1,1,10,100]
 #  }

 #  X_train, X_test, y_train, y_test,ps = train_test_split_with_indices(cancer.data, cancer.target, test_size=0.3,dtype=float)
 #  log_reg_grid_search = GridSearchCV(log_reg, parameters,
 #                     #cv=StratifiedKFold(n_splits=3,shuffle=True),
 #                     cv=ps,
 #                     scoring ='accuracy',
 #                     return_train_score=True)
 #  log_reg_grid_search.fit(X=X_train,y=y_train)
 #  print("best parameters are")
 #  print(log_reg_grid_search.best_params_ )
 #  print("the over all results are")
 #  print(log_reg_grid_search.cv_results_)

 # Here X_test, y_test is the untouched data
 # Validation data (X_val, y_val) is currently inside X_train, which will be split using PredefinedSplit inside GridSearchCV

 X_train, X_test = np.array_split(X, [len(X)])
 y_train, y_test = np.array_split(y, [len(y)])


 # The indices which have the value -1 will be kept in train.
 cut_off=int(len(X)*(1-test_size))
 train_indices = np.full((cut_off,), -1, dtype=dtype)

 # The indices which have zero or positive values, will be kept in test
 test_indices = np.full((len(X)-cut_off,), 0, dtype=dtype)
 test_fold = np.append(train_indices, test_indices)

 # print(test_fold)

 from sklearn.model_selection import PredefinedSplit
 ps = PredefinedSplit(test_fold)

 # Check how many splits will be done, based on test_fold
 # print('splits are' + str(ps.get_n_splits()))

#  for train_index, test_index in ps.split():
#      print("TRAIN:", train_index, "TEST:", test_index)

 return X_train, X_test, y_train, y_test,ps

from copy import deepcopy
def dense_arch_builder(input_size,scale_power=0,hidden_layers_num=0,repeat=0,output_size=1):
  layer_sizes=[input_size]

  if scale_power>1:
   for i in range(hidden_layers_num):
    layer_sizes.append(layer_sizes[-1]*scale_power)

  elif scale_power==1:
    for i in range(2,hidden_layers_num+2):
     layer_sizes.append(layer_sizes[0]*i)

  mirrored_layer_sizes=deepcopy(layer_sizes)
  mirrored_layer_sizes.reverse()
  mirrored_layer_sizes=mirrored_layer_sizes[1:-1]

  for i in range(repeat):
   layer_sizes.append(layer_sizes[-1])

  if output_size>0:
   layer_sizes+=mirrored_layer_sizes
   scale_downer=scale_power if scale_power>1 else 2

   while layer_sizes[-1]!=output_size:
    if layer_sizes[-1]//scale_downer>=output_size:
     layer_sizes.append(layer_sizes[-1]//scale_downer)
    else:
     layer_sizes.append(output_size)

  return layer_sizes

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
def sequential_unit_model_builder(network_type,input_shape,layer_sizes,last_layer_activation,dropout_rate=0,time_steps=0,
                                  activation=None,**kwargs):
 # if the network_type is lstm, do we even need layer_sizes or is every layer the same?
 # the reason default value of dropout_rate is zero is because dropout could actually harm the performance of the neural network, specially on val dataset
 network=[]
 # not supporting rnn, cause it's not usefull anymore
 type_activation_map={'ann':'relu','lstm':'tanh'}
 activation_kernel_initializer_map={'relu':'he_normal','tanh':'glorot_uniform'}
 layer_func_map={'ann':layers.Dense,'lstm':layers.LSTM}
 layer_func=layer_func_map[network_type]

 if activation is not None:
  type_activation_map[network_type]=activation
  if activation not in activation_kernel_initializer_map.keys():
    activation_kernel_initializer_map[activation]='he_normal'



 if network_type=='lstm':
  # here the input_shape is the number of features
  # input_shape=(time_steps,input_shape)
  kwargs['return_sequences'] = True

 network.append(layer_func(units=layer_sizes[0], activation=type_activation_map[network_type],
                           kernel_initializer=activation_kernel_initializer_map[type_activation_map[network_type]]
                           ,input_shape=input_shape,**kwargs))

 for i in range(1,len(layer_sizes)-1):
   network.append(layer_func(units=layer_sizes[i], activation=type_activation_map[network_type],
                           kernel_initializer=activation_kernel_initializer_map[type_activation_map[network_type]]
                           ,**kwargs))

 if 'return_sequences' in kwargs:
  kwargs['return_sequences'] = False


 network.append(layer_func(units=layer_sizes[-1], activation=last_layer_activation, **kwargs))

 if dropout_rate>0:
  network_with_droput=[]
  for i in range(len(network) - 1):
    network_with_droput.append(network[i])
    network_with_droput.append(layers.Dropout(rate=dropout_rate))
  network_with_droput.append(network[-1])
  network=network_with_droput

 return Sequential(network)

import matplotlib.pyplot as plt
def regression_accuracy_scatter_plot(Y_test, pred):
  plt.scatter(Y_test, pred)
  plt.xlabel("Actual Values")
  plt.ylabel("Predicted Values")
  plt.show()

from tensorflow.keras.callbacks import EarlyStopping
def sequential_model_trainer_evaluator(model,optimizer,X_train,Y_train,X_val,Y_val,X_test, Y_test,batch_size=8,epochs=20,
                             loss="mse",metrics=['mse', 'mae', 'mape'],callbacks=[EarlyStopping(monitor='loss',patience=5)]):
  # in addition to detemining the degree of which the model overfits to a certain data, batch size also determines how quickly the cost function stabalizes
  # which is not always a good thing, as that could signal the model has high bias
  model.compile(optimizer=optimizer, loss=loss,metrics=metrics)
  hist = model.fit(x=X_train, y=Y_train, batch_size=batch_size, validation_data=(X_val, Y_val),epochs=epochs,verbose=1,callbacks=callbacks)
  plt.plot(hist.history[loss])
  plt.plot(hist.history['val_'+loss])
  plt.title('Model Performance')
  plt.ylabel(loss)
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper right')
  plt.show()
  print("Evaluate on test data")
  results = model.evaluate(X_test, Y_test)
  print("test loss, test acc:", results)
  return hist

# Get the correlation between different features
import seaborn as sns
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    _ = sns.heatmap(

        df.corr(numeric_only=True),
        cmap = colormap,
        square=True,
        ax=ax,
        annot=True,
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize': 5 }
    )

    plt.title('Pearson Correlation of Features', y=1.05, size=15)

def cnn_model_builder(conv_layer,kernel_size,conv_factor,pool_layer,pool_size,input_shape,filter_sizes, activation='relu',
 kernel_initializer='he_uniform',padding='same',**kwargs):

 # by same padding, we mean zero padding that is evenly done and with stride of one keeps the output dimention the same
 cnn_model=Sequential()

 for filter_size in filter_sizes:
  for i in range(conv_factor):
   cnn_model.add(conv_layer(filters=filter_size, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer, activation=activation,
                     input_shape=input_shape[1:3]))
  cnn_model.add(pool_layer(pool_size=pool_size))

 cnn_model.add(layers.Flatten())
 cnn_model.add(layers.BatchNormalization())
 return cnn_model

def sequential_model_combiner(models):
  # Create a new Sequential model
 combined_model = Sequential()
 for model in models:
  # Add layers from the first model
  for layer in model.layers:
      combined_model.add(layer)
 return combined_model