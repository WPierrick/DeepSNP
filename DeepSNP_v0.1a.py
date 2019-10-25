from __future__ import print_function

import keras
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.utils import to_categorical
from pandas_plink import read_plink
from sklearn.preprocessing import StandardScaler

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_plink
from pandas_plink import read_plink

batch_size = 64
epochs = 30

#-----------------------------------------------------------------------------
# Load SNP and height data. Split between training and testing 
#-----------------------------------------------------------------------------

#Load train data (UKB), merge sex with genotypes, load phenotypes (normalized: (xi - min(x)) / max(x) - min(x)). Here min(x)=100 (cm)
(bim, fam, x_train) = read_plink('./01_Data_Train_UKB/00_Second_try_UKB_EURu/UKB_EURu_3k_2nd_try_noNA')
x_train = np.divide(x_train, 2)
sex_train = pd.read_csv("./01_Data_Train_UKB/00_Second_try_UKB_EURu/Sex_UKB_EURu_onehot", header=None, sep='\t')
#sex_train = sex_train.iloc[:,[3,4]]
sex_train = sex_train.transpose()
x_train = np.vstack([x_train, sex_train])
x_train = np.transpose(x_train)
#y_train = pd.read_csv('./01_Data_Train_UKB/Pheno_normalized_100cm_EURu_scaled', sep='\t', header=None, usecols=[1])
y_train = pd.read_csv('./01_Data_Train_UKB/00_Second_try_UKB_EURu/Pheno_normalized_100cm_EURu', sep='\t', header=None)

#Load test data (TOPMed), same process than UKB
(bim, fam, x_test) = read_plink('./02_Data_Test_Topmed/Topmed_subset_GIANT_UKBEURu_Hits_Height')
x_test = np.divide(x_test, 2)
sex_test = pd.read_csv("./02_Data_Test_Topmed/Sex_Topmed_21k_one_hot", header=None, sep='\t')
sex_test = sex_test.transpose()
x_test = np.vstack([x_test, sex_test])
x_test = np.transpose(x_test)
y_test = pd.read_csv('./02_Data_Test_Topmed/Pheno_normalised_100cm', sep='\t', header=None, usecols=[1])
#y_test = pd.read_csv('./02_Data_Test_Topmed/Pheno_normalised_100cm_scaled', sep='\t', header=None, usecols=[1])

x_train[np.isnan(x_train)] = 0
x_test[np.isnan(x_test)] = 0


print("x_train shape", "\t", x_train.shape, '\n', "y_train shape", "\t", y_train.shape, '\n', "x_test shape", "\t", x_test.shape, '\n', "y_test shape", "\t", y_test.shape, '\n')



print(np.mean(y_test))
print(np.mean(y_train))



#x = np.load('/Users/p.wainschtein/Documents/DeepSNP/BED_GWAS_hits_Height/Topmed_subset_GIANT_Hits_Height_genotype_and_Sex.npy')
#x_train = x[0:21519,]
#x_test = x[21520:21620,]

#y=pd.read_table("/Users/p.wainschtein/Documents/DeepSNP/BED_GWAS_hits_Height/Height_21k", sep=' ', header=None)
#y_train=y.iloc[0:21519,2]
#y_test=y.iloc[21520:21620,2]


#y_train = y_train/np.amax(y_train, axis=0)
#y_test = y_test/np.amax(y_test, axis=0)
#print(x_train.shape, 'train samples')
#print(x_test.shape, 'test samples')

#-----------------------------------------------------------------------------
# Define custom loss functions for regression in Keras 
#-----------------------------------------------------------------------------

# root mean squared error (rmse) for regression
def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# mean squared error (mse) for regression
def mse(y_true, y_pred):
    return backend.mean(backend.square(y_pred - y_true), axis=-1)

# coefficient of determination (R^2) for regression
def r_square(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))

def r_square_loss(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return 1 - ( 1 - SS_res/(SS_tot + K.epsilon()))


#-----------------------------------------------------------------------------
# Define Keras MLP model (Sequencial  = multilayer perceptron)
#-----------------------------------------------------------------------------

model = Sequential()
model.add(Dense(20, activation='relu', kernel_initializer='normal', input_shape=(3164,)))
model.add(Dropout(0.01))
model.add(Dense(20, activation='relu', kernel_initializer='normal'))
model.add(Dense(20, activation='tanh', kernel_initializer='normal'))
model.add(Dense(1))
model.summary()

#opt= keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#model.compile(optimizer = opt, loss="mse", metrics=["mse", r_square])
model.compile(optimizer = opt, loss="mse", metrics=["mse"])


#-----------------------------------------------------------------------------
# Run the model
#-----------------------------------------------------------------------------

result = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1,
                    validation_data=(x_train, y_train))

#result = model.fit(x_train, y_train, epochs=240, batch_size=5, validation_data=(x_test, y_test), callbacks=[earlystopping])
y_predict = model.predict(x_test)
#y_test2 = np.expand_dims(y_test, axis=1)

y_pred_train = model.predict(x_train)

y_test = y_test.values

#plt.scatter(y_predict.T, y_test2.T)
#plt.title('Correlation predicted / test phenotype')
#plt.ylabel('predicted')
#plt.xlabel('test')
#plt.show()
print('Correlation between predicted and test datasets: ', np.corrcoef(y_predict.T, y_test.T)[1][0])


#print('Correlation between predicted and train datasets: ', np.corrcoef(y_pred_train.T, y_train.T)[1][0])

score_test = model.evaluate(x_test, y_test, verbose=0)
'''
print('Evaluate on test data')
print('Test loss:', score_test[0])
print('Test MSE:', score_test[1])
print('Test R^2:', score_test[2])

score_train = model.evaluate(x_train, y_train, verbose=0)
print('Evaluate on train data')
print('Test loss:', score_train[0])
print('Test MSE:', score_train[1])
print('Test R^2:', score_train[2])
'''

#-----------------------------------------------------------------------------
# Plot learning curves including R^2 and RMSE
#-----------------------------------------------------------------------------

'''
#plt.plot(result.history['val_mse'])
plt.plot(result.history['mean_squared_error'])
plt.plot(result.history['val_mean_squared_error'])
plt.title('model mse')
plt.ylabel('mse')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# plot training curve for R^2 (beware of scale, starts very low negative)
plt.plot(result.history['r_square'])
plt.plot(result.history['val_r_square'])
plt.title('model R^2')
plt.ylabel('R^2')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
'''


######### OLD CODE ###########

'''
#With MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_train)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
'''
'''
#With made up dataset
x_train = np.random.randint(3, size=(5000,10000))
x_test = np.random.randint(3, size=(100,10000))

y_train = np.random.normal(175, 6, 5000)
y_test = np.random.normal(175, 6, 100)

x_train = x_train/np.amax(x_train, axis=0)
x_test = x_test/np.amax(x_test, axis=0)
'''
'''
# the data, split between train and test sets
(bim, fam, G) = read_plink('/Users/p.wainschtein/Documents/DeepSNP/BED_GWAS_hits_Height/Topmed_subset_GIANT_Hits_Height')
g=G.compute()
g2 = np.transpose(g)
sex = np.loadtxt('/Users/p.wainschtein/Documents/DeepSNP/BED_GWAS_hits_Height/Sex_21k')
sex = sex[:, None]
g2 = np.hstack((g2, sex))
x_train = g2[0:21519,]
x_train = np.nan_to_num(x_train)
x_test = g2[21520:21620,]
x_test = np.nan_to_num(x_test)
'''
'''
sc = StandardScaler()

y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)

x_test = sc.fit_transform(x_test)
x_train = sc.fit_transform(x_train)
y_train = sc.fit_transform(y_train)
y_test = sc.fit_transform(y_test)
'''
#x_test = x_test/np.amax(x_test, axis=0)
#x_train = x_train/np.amax(x_train, axis=0)


'''
Full feature model
model = Sequential()
model.add(Dense(200, activation='tanh', kernel_initializer=keras.initializers.Constant(0.1), bias_initializer=keras.initializers.Constant(0.1), input_shape=(3164,)))
model.add(Dropout(0.1))
model.add(Dense(100, activation='tanh', bias_initializer=keras.initializers.Constant(0.1)))
model.add(Dropout(0.1))
model.add(Dense(10, activation='tanh', bias_initializer=keras.initializers.Constant(0.1)))
model.add(Dropout(0.1))
#model.add(Dense(64, kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dense(5, activation='tanh',  bias_initializer=keras.initializers.Constant(0.1)))
model.add(Dropout(0.1))
model.add(Dense(1, activation='tanh'))
model.summary()

#model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

#opt = SGD(lr=0.01)
#model.compile(loss = "categorical_crossentropy", optimizer = opt)
#model.compile(optimizer='rmsprop', loss='mse', metrics=["mean_squared_error", rmse, r_square])
'''


'''
# print the linear regression and display datapoints
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(y_test.values.reshape(-1,1), y_pred)  
y_fit = regressor.predict(y_pred) 

reg_intercept = round(regressor.intercept_[0],4)
reg_coef = round(regressor.coef_.flatten()[0],4)
reg_label = "y = " + str(reg_intercept) + "*x +" + str(reg_coef)

plt.scatter(y_test, y_pred, color='blue', label= 'data')
plt.plot(y_pred, y_fit, color='red', linewidth=2, label = 'Linear regression\n'+reg_label) 
plt.title('Linear Regression')
plt.legend()
plt.xlabel('observed')
plt.ylabel('predicted')
plt.show()

#-----------------------------------------------------------------------------
# print statistical figures of merit
#-----------------------------------------------------------------------------

import sklearn.metrics, math
print("\n")
print("Mean absolute error (MAE):      %f" % sklearn.metrics.mean_absolute_error(y_test,y_pred))
print("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(y_test,y_pred))
print("Root mean squared error (RMSE): %f" % math.sqrt(sklearn.metrics.mean_squared_error(y_test,y_pred)))
print("R square (R^2):                 %f" % sklearn.metrics.r2_score(y_test,y_pred))

plot_model(history, to_file='model.png')

# Plot training & validation accuracy values
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['r_square'])
plt.plot(history.history['val_r_square'])
plt.title('Model r_square')
plt.ylabel('R_square')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
'''
