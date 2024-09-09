import stat
import os
def python_mkdir(parent_dir,new_dir,exist_ok=False):
 st = os.stat(parent_dir)
 original_permissions = stat.S_IMODE(st.st_mode)
 try:
  if exist_ok:
        os.makedirs(new_dir, exist_ok=True)
  elif not os.path.exists(new_dir):
    os.makedirs(new_dir)
  else:
    i=0
    while os.path.exists(new_dir+str(i)):
      i+=1
      new_dir=new_dir+str(i)
    os.makedirs(new_dir)
 except OSError as e:
   print(f"Error: {e}")
 finally:
     os.chmod(parent_dir, original_permissions)  # Revert to original permissions
     print(f"Permissions reverted to {oct(original_permissions)} for {parent_dir}")
 return new_dir  
 
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
def sequential_unit_model_builder(name,network_type,input_shape,layer_sizes,last_layer_activation,dropout_rate=0,time_steps=0,
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

 return Sequential(network,name=name)



from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
def sequential_model_trainer_evaluator(model,optimizer,X_train,Y_train,X_val,Y_val,X_test, Y_test,batch_size=8,epochs=20,
                             loss="mse",metrics=['mse', 'mae', 'mape'],format='keras',callbacks=None):
  
  # note: this function is mainly for small datasets, as bigger datasets need a data loader
  
  # Loss is typically preferred for its sensitivity and ability to reflect small changes in model learning, especially in imbalanced or regression tasks
  # Accuracy is more intuitive and can be more directly related to model performance in balanced classification tasks
								  
  # in addition to detemining the degree of which the model overfits to a certain data, batch size also determines how quickly the cost function stabalizes
  # which is not always a good thing, as that could signal the model has high bias
  parent_dir = os.getcwd()
  export_path = os.path.join(parent_dir, model.name)
  export_path=python_mkdir(parent_dir,export_path)

  if callbacks is None:
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
               tf.keras.callbacks.ModelCheckpoint(filepath=export_path+model.name+'.'+format, monitor='val_loss', save_best_only=True)]

  model.compile(optimizer=optimizer, loss=loss,metrics=metrics)
  hist = model.fit(x=X_train, y=Y_train, batch_size=batch_size, validation_data=(X_val, Y_val),epochs=epochs,verbose=1,callbacks=callbacks)
  plt.plot(hist.history[loss])
  plt.plot(hist.history['val_'+loss])
  plt.title('Model Performance')
  plt.ylabel(loss)
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper right')
  plt.show()
  plt.savefig(export_path+'Model_Loss.png')
  results = model.evaluate(X_test, Y_test)
  results_keys=[loss,*metrics]
  results_dict={}
  for i in range(len(results_keys)):
   results_dict[results_keys[i]]=results[i]
  print(f"evaluation results are {results_dict}")                    
  return hist,results_dict

# Get the correlation between different features
import seaborn as sns
import matplotlib.pyplot as plt
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

def cnn_model_builder(name,conv_layer,kernel_size,conv_factor,pool_layer,pool_size,input_shape,filter_sizes, activation='relu',
 kernel_initializer='he_uniform',padding='same',**kwargs):

 # by same padding, we mean zero padding that is evenly done and with stride of one keeps the output dimention the same
 cnn_model=Sequential(name=name)

 for i in range(len(filter_sizes)):
   for j in range(conv_factor):
    if i==0:
     cnn_model.add(conv_layer(filters=filter_sizes[i], kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer, activation=activation,
                     input_shape=input_shape))
    else:
     cnn_model.add(conv_layer(filters=filter_sizes[i], kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer, activation=activation,
                     input_shape=input_shape))
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

import math
def data_visualizer(df, heatmap=False, displot_kind=None):

  if not isinstance(df, pd.DataFrame):df=pd.DataFrame(df)

  print('data info:')
  print(df)
  print(df.info())
  print(df.describe())
  print(df.isna().sum())

  if heatmap:
   correlation_heatmap(df)

  if displot_kind is not None:
  #  df.hist(bins=50, figsize=(20,15))
  #  plt.show()
   num_columns = int(math.sqrt(len(df.columns)))  # Define the number of columns for the grid
   num_rows = len(df.columns) // num_columns + (len(df.columns) % num_columns > 0)  # Calculate the number of rows needed
 
   fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 10))
 
   # Flatten the axes array for easy iteration
   axes = axes.flatten()
 
   # Plotting each attribute in the grid
   for i, column in enumerate(df.columns):
       sns.histplot(df[column], kde=displot_kind, ax=axes[i])
       axes[i].set_title(column)
 
   # Remove any unused subplots
   for j in range(i + 1, len(axes)):
       fig.delaxes(axes[j])
 
   plt.tight_layout()
   plt.show()
