# -*- coding: utf-8 -*-
#https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
#https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
#https://datascience.stackexchange.com/questions/43191/validation-loss-is-not-decreasing
#https://stats.stackexchange.com/questions/425610/why-massive-random-spikes-of-validation-loss
"""
Created on Wed Nov  2 20:54:46 2022

@author: Benjamín Martín Gómez
"""
import tensorflow as tf
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import dataNormalization as dn
from matplotlib import pyplot
import math
import threading
import time
from keras.optimizers import Adam

def predict(model, in_seq, in_seq_test_norm, in_seq_truth, recurrent_forecast, normalization, minimo, maximo, inicio_val, final_val, timesteps_past, n_subseqs, n_steps, timesteps_future_to_predict, timesteps_future, theta_series_v2, theta, CNN):
    if recurrent_forecast == 1:
        recurrent_steps = math.ceil(timesteps_future_to_predict/timesteps_future) #Número de veces que se hace la predicción hasta llegar al valor de multistep
        for i in range(recurrent_steps):
            yhat = model.predict(in_seq_test_norm, verbose=0) #1 x timesteps_future
            yhat = yhat.reshape(yhat.shape[1],)
            
            if normalization == 0:
                seq_prediction = dn.un_normalizeData_MinMax(yhat, minimo, maximo)
            if normalization == 1:
                seq_prediction = dn.un_normalizeData_tanh(yhat, minimo, maximo)
            if normalization == 2:
                seq_prediction = dn.un_normalizeData_zscore(yhat, minimo, maximo)
            
            if i == 0:
                real = np.concatenate((in_seq, in_seq_truth)); prediction = np.concatenate((in_seq, seq_prediction))
            else:
                #Truth:
                inicio_val = inicio_val+timesteps_future; final_val = final_val+timesteps_future;
                in_seq_truth = theta_series_v2[semana][theta][inicio_val:final_val+1]
                real = np.concatenate((real, in_seq_truth)); prediction = np.concatenate((prediction, seq_prediction))
                
            #Usamos la predicción como test para la siguiente predicción:
            in_seq_test = prediction[-timesteps_past:]
            
            if normalization == 0:
                in_seq_test_norm = dn.normalizeData_MinMax_using_scalerData(in_seq_test, minimo, maximo)
            if normalization == 1:
                in_seq_test_norm = dn.normalizeData_tanh_using_scalerData(in_seq_test, minimo, maximo)
            if normalization == 2:
                in_seq_test_norm = dn.normalizeData_zscore_using_scalerData(in_seq_test, minimo, maximo)
            
            
            
            #Para LSTM CNN:
            if CNN == 1:
                in_seq_test_norm = in_seq_test_norm.reshape((1, n_subseqs, n_steps, n_features))
            #Para LSTM simple:
            if CNN == 0:
                in_seq_test_norm = in_seq_test_norm.reshape((1, timesteps_past, n_features))
                
            
            
        
        fig, ax = pyplot.subplots(figsize=(8, 6)) #New figure
        ax.plot(real);
        ax.plot(prediction);
        pyplot.ylim(0.8*min(real), 1.2*max(real))
    
    else:
        #Prediction:
        yhat = model.predict(in_seq_test_norm, verbose=0) #1 x timesteps_future
        yhat = yhat.reshape(yhat.shape[1],)
        
        if normalization == 0:
            seq_prediction = dn.un_normalizeData_MinMax(yhat, minimo, maximo)
        if normalization == 1:
            seq_prediction = dn.un_normalizeData_tanh(yhat, minimo, maximo)
        if normalization == 2:
            seq_prediction = dn.un_normalizeData_zscore(yhat, minimo, maximo)
            
        real = np.concatenate((in_seq, in_seq_truth)); prediction = np.concatenate((in_seq, seq_prediction))
        
        #Por si queremos hacer el plot con el entrenamiento y la predicción normalizados:
        #in_seq1_truth_norm = in_seq1_truth_norm.reshape(in_seq1_truth_norm.shape[1],)
        #real1 = np.concatenate((in_seq1_norm, in_seq1_truth_norm)); prediction1 = np.concatenate((in_seq1_norm, yhat1))
        
        fig, ax = pyplot.subplots(figsize=(8, 6)) #New figure
        ax.plot(real);
        ax.plot(prediction);
        
    
    predicted_theta = prediction[-1]
    return predicted_theta
        

def create_model(theta, time_past, time_subseqs, T_train, time_neighbour_points, normalization, CNN, theta_series_v2):
        kernel_size = round(time_neighbour_points/diezmado)

        T_train = round(T_train/diezmado) #segundos. Tamaño de ventana de tiempo de entrenamiento. En multiescala de 180segs: un buen ejemplo es 100 + batch size = 1 + 300 unidades + 35 epochs
        timesteps_past = round(time_past/diezmado) #segundos. En multiescala de 180segs: un buen ejemplo es 5

        n_steps = round(time_subseqs/diezmado) #Número de puntos de la subsecuencia

        rest = divmod(timesteps_past, n_steps)
        if rest[1] is not 0:
            n_steps = get_closer_subseq_nsteps(timesteps_past, n_steps)
        n_subseqs = int(timesteps_past/n_steps) #Número de subsecuencias

        #https://datascience.stackexchange.com/questions/43191/validation-loss-is-not-decreasing
        #https://stackoverflow.com/questions/61287322/validation-loss-sometimes-spiking En este enlace se explica que un tamaño de batch más pequeño produce spikes en las pérdidas de validación
            #Train:
        inicio_train = final - T_train + 1; #. En multiescala de 180segs: un buen ejemplo es 500
        
        #in_seq = theta_series_v2[semana][theta][inicio_train:final+1] #Serie temporal theta_i
        in_seq = np.concatenate((theta_series_v2[7][theta][inicio_train:final+T_train+1], theta_series_v2[8][theta][inicio_train:final+T_train+1], theta_series_v2[semana][theta][inicio_train:final+1]))
        #Drop nan:
        in_seq = in_seq[np.logical_not(np.isnan(in_seq))]
        
        if normalization == 0:
            in_seq_norm, minimo, maximo = dn.normalizeData_MinMax(in_seq)
        if normalization == 1:
            in_seq_norm, minimo, maximo = dn.normalizeData_tanh(in_seq)
        if normalization == 2:
            in_seq_norm, minimo, maximo = dn.normalizeData_zscore(in_seq)

        X, y = split_sequence(in_seq_norm, timesteps_past, timesteps_future)

        """
        #Análisis PCA:-----------------------------------------------------------------
        from sklearn.decomposition import PCA
        # fit PCA algorithm to your data
        #flip X1:
        X1 = flip(X1) #Hacemos un flip porque así las primeras características son las de tiempos más recientes, que tienen más información
        pca = PCA().fit(X1)
        cumulative_explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        # plot the variance ratio according to the number of components
        pyplot.plot(cumulative_explained_variance_ratio)
        pyplot.xlabel('number of components')
        pyplot.ylabel('cumulative explained variance');
        #Find number of components that retain the 'X' % of information:
        percent = 0.9997
        i=0
        n_components = X1.shape[1]
        for e in cumulative_explained_variance_ratio:
            if e > percent:
                n_components = i
                break
            i=i+1
        print('Number of past steps reduced by PCA to '+str(n_components)+'\n')
        #Apply PCA to data:
        model = PCA(n_components=n_components)
        model.fit(X1)
        X1 = model.transform(X1)
        fig, ax = pyplot.subplots(figsize=(8, 6)) #New figure
        pyplot.scatter(X1[:, 0], X1[:, 1], alpha=0.2)
        #unflip:
        X1 = flip(X1)
        #------------------------------------------------------------------------------
        """

        if CNN == 1: #Para LSTM CNN:
            X = X.reshape((X.shape[0], n_subseqs, n_steps, n_features));
        if CNN == 0: #Para LSTM simple:
            X = X.reshape((X.shape[0], X.shape[1], n_features)); y = y.reshape((y.shape[0], y.shape[1], n_features))
        
        
            #Test:
        inicio_test = final - timesteps_past + 1;
        in_seq_test = theta_series_v2[semana][theta][inicio_test:final+1] #Serie temporal theta0
        if normalization == 0:
            in_seq_test_norm = dn.normalizeData_MinMax_using_scalerData(in_seq_test, minimo, maximo)
        if normalization == 1:
            in_seq_test_norm = dn.normalizeData_tanh_using_scalerData(in_seq_test, minimo, maximo)
        if normalization == 2:
            in_seq_test_norm = dn.normalizeData_zscore_using_scalerData(in_seq_test, minimo, maximo)

        """
        #Análisis PCA:-----------------------------------------------------------------
        in_seq1_test_norm_for_pca = in_seq1_test_norm.reshape((1, in_seq1_test_norm.shape[0]))
        in_seq1_test_norm = model.transform(in_seq1_test_norm_for_pca)
        in_seq1_test_norm = in_seq1_test_norm.reshape((in_seq1_test_norm.shape[1], ))
        #------------------------------------------------------------------------------
        """
        #Para LSTM CNN:
        if CNN == 1:
            in_seq_test_norm = in_seq_test_norm.reshape((1, n_subseqs, n_steps, n_features))
        #Para LSTM simple:
        if CNN == 0:
            in_seq_test_norm = in_seq_test_norm.reshape((1, timesteps_past, n_features))

            #Truth (validation)
        inicio_val = (final + 1); final_val = inicio_val+timesteps_future-1;
        in_seq_truth = theta_series_v2[semana][theta][inicio_val:final_val+1] #Serie temporal theta0
        if normalization == 0:
            in_seq_truth_norm = dn.normalizeData_MinMax_using_scalerData(in_seq_truth, minimo, maximo)
        if normalization == 1:
            in_seq_truth_norm = dn.normalizeData_tanh_using_scalerData(in_seq_truth, minimo, maximo)
        if normalization == 2:
            in_seq_truth_norm = dn.normalizeData_zscore_using_scalerData(in_seq_truth, minimo, maximo)

        #Para LSTM CNN:
        if CNN == 1:
            in_seq_truth_norm = in_seq_truth_norm.reshape((1, in_seq_truth_norm.shape[0], 1))
        #Para LSTM simple:
        if CNN == 0:
            in_seq_truth_norm = in_seq_truth_norm.reshape((1, in_seq_truth_norm.shape[0], n_features))

        
        #CNN LSTM:
        if CNN == 1:
            model = Sequential()
            #ENCODER:
            model.add(TimeDistributed(Conv1D(filters=16, kernel_size=kernel_size, activation='relu'), input_shape=(None, X.shape[2], n_features)))
            model.add(TimeDistributed(Conv1D(filters=8, kernel_size=kernel_size, activation='relu')))
            #model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
            model.add(TimeDistributed(Flatten()))
            model.add(LSTM(70, activation='relu'))
            #SIZE FIX:
            model.add(RepeatVector(timesteps_future))
            #DECODER:
            model.add(LSTM(70, activation='relu', return_sequences=True))
            model.add(TimeDistributed(Dense(1)))
            #optimizer = Adam(learning_rate=0.001)
            #model.compile(optimizer=optimizer, loss='mse')
            model.compile(optimizer='adam', loss='mse')
            """
            model = Sequential()
            model.add(Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=(X.shape[2], n_features)))
            model.add(Conv1D(filters=32, kernel_size=1, activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(RepeatVector(timesteps_future))
            model.add(LSTM(200, activation='relu', return_sequences=True))
            model.add(TimeDistributed(Dense(100, activation='relu')))
            model.add(TimeDistributed(Dense(1)))
            model.compile(optimizer='adam', loss='mse')
            """
        
        #LSTM Encoder Decoder simple:
        if CNN == 0:
            model = Sequential()
            model.add(LSTM(70, activation='relu', input_shape=(timesteps_past, n_features)))
            model.add(RepeatVector(timesteps_future))
            model.add(LSTM(70, activation='relu', return_sequences=True))
            model.add(TimeDistributed(Dense(1)))
            model.compile(optimizer='adam', loss='mse')
        
        
        return X, y, in_seq, in_seq_truth, in_seq_test_norm, in_seq_truth_norm, model, inicio_val, final_val, timesteps_past, minimo, maximo, n_steps, n_subseqs


def flip(X):
    #X = [[1 2 3 4], [5 6 7 8]] (for example)
    #result = [[4 3 2 1], [8 7 6 5]]
    for i in range(X.shape[0]): #for each list in X ([1, 2, 3, 4] for example)
        #flip it:
        X[i] = X[i][::-1]
    
    return X

def get_closer_subseq_nsteps(timesteps_past, subseqpoints):
    for i in range(1, timesteps_past-subseqpoints):
        n_steps_result_asc = subseqpoints+i
        asc_result = divmod(timesteps_past, n_steps_result_asc)
        if asc_result[1] == 0:
            return n_steps_result_asc
           
        n_steps_result_desc = subseqpoints-i
        desc_result = divmod(timesteps_past, n_steps_result_desc)
        if desc_result[1] == 0:
            return n_steps_result_desc
        
    return timesteps_past
    
# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def trainLSTM(list, X, y, epochs, verbose, test_norm, truth_norm, model, ide):

    class haltCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('val_loss') <= 1e-18): #Para evitar overfitting
                print("Value reached\n\n")
                self.model.stop_training = True
    trainingStopCallback = haltCallback()

    history = model.fit(X, y, epochs=epochs, verbose=verbose, validation_data=(test_norm, truth_norm), callbacks=[trainingStopCallback]) #callbacks=[trainingStopCallback]
    list[ide] = history
    
    """
    if ide == 5:
        fig, ax = pyplot.subplots(figsize=(8, 6))
        pyplot.plot(history.history['loss'])
        pyplot.plot(history.history['val_loss'])
        pyplot.title('model train vs validation loss')
        pyplot.ylabel('loss')
        pyplot.xlabel('epoch')
        pyplot.legend(['train', 'validation'], loc='upper right')
        pyplot.ylim((0, 1))
        pyplot.show()
    """
    
    
    
    
#------------------------------------------------------------------------------
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
Tsventana = 30*60 #Tamaño de ventana que se usó para sacar los coeficientes y parámetros alpha-stable
n = 7; #Grado de la regresión polinómica que se usó cuando se obtuvieron estos parámetros theta

#Importación de datos:
pyplot.close('all')
timesteps_future = 1;
timesteps_future_recurrent = 1;
diezmado = 180; #[s]
recurrent_forecast = 0; #0: no se desea predicción recurrente. 1: se desea predicción recurrente (se adivina un punto y se usa para la siguiente predicción y así hasta completar los multistep puntos)
normalization = 2; #0: MinMax. 1: tanh. 2: zscore
CNN = 0; #1: CNN + LSTM. 0: LSTM simple
velocity = 0; #Indica si se desea trabajar con la derivada de la evolución de los parámetros theta en lugar de usar la evolución de los parámetros theta. 1: Usar velocidad (derivada). 2: no usarla

#Si por ejemplo diezmado = 1s y timesteps_future = 1, queremos evaluar la próxima ventana que está a diezmado*timesteps_future de distancia temporal (en este ejemplo, a 1*1s de distancia)
#Si diezmado = 60s y timesteps_future = 1, entonces queremos hacer una predicción unistep (porque timestep future es 1) usando un diezmado cada 60s, así que el punto que se va a predecir es la próxima ventana que dista 60*1 segs de la última ventana conocida
#Si diezmado = 30s y timesteps_future = 2, es el mismo caso que antes, solo que ahora lo vamos a hacer como una predicción multistep: muestreamos la señal theta cada 30s y predecimos los próximos dos puntos, que en realidad es predecir los coeficientes en la ventana que dista 30*2 = 60s de la última ventana conocida de la que se tienen datos
#Por tanto, llamaremos alcance al producto diezmado*timesteps_future:
scope = timesteps_future*diezmado
print('Model forecast coefficients for the next '+str(scope)+' seconds\n')

filename_thetaParams = "DataBase/TP30_7.txt";
filename_network_traffic_series = 'DataBase/All_series.txt';
all_series = np.loadtxt(filename_network_traffic_series, delimiter=',')
#Fichero con el siguiente formato:
n_coefs = n+1
thetaParams = np.loadtxt(filename_thetaParams, delimiter=',')

n_series = int(thetaParams.shape[1]/n_coefs)
n_ventanas = math.ceil(thetaParams.shape[0]/diezmado)
if velocity == 0:
    theta_series = np.zeros((n_coefs, n_series, n_ventanas))
if velocity == 1:
    theta_series = np.zeros((n_coefs, n_series, n_ventanas-1))
#theta_series[0] = paquete con las series temporales de theta 0 ordenadas de arriba a abajo en orden semanal. Así, la primera fila del paquete theta_series[0] es la serie temporal de theta0 de la última semana
    #theta_series[0][0] = serie theta 0 más antigua (marzo semana 3)
    #theta_series[0][1] = serie theta 0 de la semana siguiente a la más antigua
    #...
#theta_series[1] = paquete con las series temporales de theta 1...

#Orden semanal:
"""
0 = march_week3
1 = march_week4
2 = march_week5

3 = april_week2
4 = april_week4

5 = may_week1
6 = may_week3

7 = june_week1
8 = june_week2
9 = june_week3

10 = july_week1
"""

for c in range(1, n_coefs+1): #Por cada coeficiente (hay n+1 coeficientes. Recorremos de 1 a n_coefs. Ponemos +1 porque en python el último valor no cuenta)
    for serie in range(1, n_series+1): #Por cada serie
        theta_evolution = thetaParams[::diezmado, (serie-1)*n_coefs+c-1]
        if velocity == 0:
            theta_series[c-1][serie-1] = theta_evolution
        if velocity == 1:
            theta_series[c-1][serie-1] = np.diff(theta_evolution)/diezmado

#Por si es de utilidad, vamos a crear el array de parámetros theta en otro formato de almacenamiento:
if velocity == 0:
    theta_series_v2 = np.zeros((n_series, n_coefs, n_ventanas))
if velocity == 1:
    theta_series_v2 = np.zeros((n_series, n_coefs, n_ventanas-1))
#theta_series_v2[0] = Parámetros theta de la serie temporal de marzo week3
    #theta_series_v2[0][0] = Parámetro theta0 de la serie temporal de marzo week3
    #theta_series_v2[0][1] = Parámetro theta1 de la serie temporal de marzo week3
    #...
#theta_series_v2[1] = Parámetros theta de la serie temporal de marzo week4
#theta_series_v2[2] = Parámetros theta de la serie temporal de marzo week5
#...

for serie in range(1, n_series+1):
    for c in range(1, n_coefs+1):
        theta_evolution = thetaParams[::diezmado, (serie-1)*n_coefs+c-1]
        if velocity == 0:
            theta_series_v2[serie-1][c-1] = theta_evolution
        if velocity == 1:
            theta_series_v2[serie-1][c-1] = np.diff(theta_evolution)/diezmado

#------------------------------------------------------------------------------
#Prueba de entrenamiento y predicción de parámetros theta:
#Aquí vamos a considerar solo una de las semanas. Usando la información en los momentos previos (no de las semanas anteriores) intentaremos predecir nuevas tendencias:
semana = 9 #Indexada desde el 0 incluido
time_serie = all_series[semana]

#Train:
    #Buenos resultados: past = 6 (o 5) puntos, future = 1, 50 epoch, 100 unidades LSTM, T_train = 5 mins = 300s
#No cogeremos toda la serie temporal, sino solo 15 mins cualesquiera
#inicio es el tiempo en segundos (incluido) a partir del cual se cogen valores. Se empieza a indexar desde el 0 ([0] = primer segundo)
#final es el tiempo en segundos (incluido) hasta el cual se cogen valores.
tiempo_final = 27849; #Segundo de las series theta_i final del que se conocen datos (en segundos, desde las 00:00:00 del lunes)
final = round(tiempo_final/diezmado); #Punto final
tiempo_final = final*diezmado;

hora_final = tiempo_final/3600
minuto_final = (hora_final-int(hora_final))*60
segundo_final = (minuto_final-int(minuto_final))*60

print('Última ventana conocida: Lunes [00:00:00] + ['+str(int(hora_final))+'h, '+str(int(minuto_final))+'min, '+str(int(segundo_final))+'s]\n')


n_features = 1 #Número de series temporales por cada LSTM (1 por cada LSTM)
history_list = [0]*(n+1)

timesteps_future_to_predict = 0
if recurrent_forecast == 1:
    timesteps_future_to_predict = timesteps_future #Timesteps futuros que puso el usuario en su momento (arriba)
    timesteps_future = timesteps_future_recurrent #Timesteps future que usará la red neuronal. <------------------------------------------------------------ SE PUEDE MODIFICAR POR EL USUARIO
    resultdivmod = divmod(timesteps_future_to_predict, timesteps_future)
    if resultdivmod[1] is not 0:
        print('WARNING: El usuario ha introducido un valor de timesteps_future para la predicción recurrente que no es divisible con el número de timesteps futuros que se quiere predecir. Se van a predecir algunos puntos más de los indicados\n')

#Window_Selected = time_serie[(final-1)*granuralidad_deteccion+1+granuralidad_deteccion -(Tsventana-1):(final-1)*granuralidad_deteccion+1+granuralidad_deteccion+1]
window_selected_index = round((tiempo_final+scope)/diezmado)
Window_Selected = time_serie[window_selected_index*diezmado:window_selected_index*diezmado+Tsventana]

#------------------------------------------------------------------------------
#Estudio de clustering con los coeficientes del polinomio:
from sklearn.decomposition import PCA
# fit PCA algorithm to your data
X_data = np.transpose(theta_series_v2[semana][:])
#Drop rows with NaN values:
nan_rows = np.where(np.isnan(X_data).any(axis=1))[0]
X_data = np.delete(X_data, nan_rows, axis=0)
X_data_norm = X_data
#Normalize each column:
for i in range(X_data.shape[1]):
    X_data_norm[:, i] = dn.normalizeData_zscore(X_data[:, i])[0]

pca = PCA().fit(X_data_norm)
cumulative_explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
# plot the variance ratio according to the number of components
pyplot.plot(cumulative_explained_variance_ratio)
pyplot.xlabel('number of components')
pyplot.ylabel('cumulative explained variance');
#Find number of components that retain the 'X' % of information:
percent = 0.9997
i=0
n_components = X_data_norm.shape[1]
for e in cumulative_explained_variance_ratio:
    if e > percent:
        n_components = i
        break
    i=i+1
print('Number of past steps reduced by PCA to '+str(n_components)+'\n')
#Apply PCA to data:
model = PCA(n_components=n_components)
model.fit(X_data_norm)
X_data_norm = model.transform(X_data_norm)
#------------------------------------------------------------------------------










































#Modelos:
#THETA0:--------------------------------------------------------------------------------------------------
#theta = 0 #0 = theta0, 1 = theta1, 2 = theta2...
time_past0 = (360/diezmado)*diezmado #Si es demasiado bajo, empeoran las val_loss. Si es demasiado alto, empeoran también. 300
time_subseqs0 = time_past0
T_train0 = 3000*diezmado #Cuanto más alto, mejoran las val_loss bastante. 1700*diezmado | tiempo_final-3600 | (final-100)*diezmado
time_neighbour_points0 = 1*diezmado; #Tiempo [s] de análisis que se usa en la capa convolucional para evaluar muestras vecinas por cada subsecuencia

X0, y0, in_seq0, in_seq0_truth, in_seq0_test_norm, in_seq0_truth_norm, model0, inicio_val0, final_val0, timesteps_past, minimo0, maximo0, n_steps0, n_subseqs0 = create_model(theta=0, time_past=time_past0, time_subseqs=time_subseqs0, T_train=T_train0, time_neighbour_points=time_neighbour_points0, normalization=normalization, CNN=CNN, theta_series_v2=theta_series_v2)
X1, y1, in_seq1, in_seq1_truth, in_seq1_test_norm, in_seq1_truth_norm, model1, inicio_val1, final_val1, timesteps_past, minimo1, maximo1, n_steps1, n_subseqs1 = create_model(theta=1, time_past=time_past0, time_subseqs=time_subseqs0, T_train=T_train0, time_neighbour_points=time_neighbour_points0, normalization=normalization, CNN=CNN, theta_series_v2=theta_series_v2)
X2, y2, in_seq2, in_seq2_truth, in_seq2_test_norm, in_seq2_truth_norm, model2, inicio_val2, final_val2, timesteps_past, minimo2, maximo2, n_steps2, n_subseqs2 = create_model(theta=2, time_past=time_past0, time_subseqs=time_subseqs0, T_train=T_train0, time_neighbour_points=time_neighbour_points0, normalization=normalization, CNN=CNN, theta_series_v2=theta_series_v2)
X3, y3, in_seq3, in_seq3_truth, in_seq3_test_norm, in_seq3_truth_norm, model3, inicio_val3, final_val3, timesteps_past, minimo3, maximo3, n_steps3, n_subseqs3 = create_model(theta=3, time_past=time_past0, time_subseqs=time_subseqs0, T_train=T_train0, time_neighbour_points=time_neighbour_points0, normalization=normalization, CNN=CNN, theta_series_v2=theta_series_v2)
X4, y4, in_seq4, in_seq4_truth, in_seq4_test_norm, in_seq4_truth_norm, model4, inicio_val4, final_val4, timesteps_past, minimo4, maximo4, n_steps4, n_subseqs4 = create_model(theta=4, time_past=time_past0, time_subseqs=time_subseqs0, T_train=T_train0, time_neighbour_points=time_neighbour_points0, normalization=normalization, CNN=CNN, theta_series_v2=theta_series_v2)
X5, y5, in_seq5, in_seq5_truth, in_seq5_test_norm, in_seq5_truth_norm, model5, inicio_val5, final_val5, timesteps_past, minimo5, maximo5, n_steps5, n_subseqs5 = create_model(theta=5, time_past=time_past0, time_subseqs=time_subseqs0, T_train=T_train0, time_neighbour_points=time_neighbour_points0, normalization=normalization, CNN=CNN, theta_series_v2=theta_series_v2)
X6, y6, in_seq6, in_seq6_truth, in_seq6_test_norm, in_seq6_truth_norm, model6, inicio_val6, final_val6, timesteps_past, minimo6, maximo6, n_steps6, n_subseqs6 = create_model(theta=6, time_past=time_past0, time_subseqs=time_subseqs0, T_train=T_train0, time_neighbour_points=time_neighbour_points0, normalization=normalization, CNN=CNN, theta_series_v2=theta_series_v2)
X7, y7, in_seq7, in_seq7_truth, in_seq7_test_norm, in_seq7_truth_norm, model7, inicio_val7, final_val7, timesteps_past, minimo7, maximo7, n_steps7, n_subseqs7 = create_model(theta=7, time_past=time_past0, time_subseqs=time_subseqs0, T_train=T_train0, time_neighbour_points=time_neighbour_points0, normalization=normalization, CNN=CNN, theta_series_v2=theta_series_v2)

epoch = 100
my_thread0 = threading.Thread(target=trainLSTM, args=(history_list, X0, y0, epoch, 0, in_seq0_test_norm, in_seq0_truth_norm, model0, 0))
my_thread1 = threading.Thread(target=trainLSTM, args=(history_list, X1, y1, epoch, 1, in_seq1_test_norm, in_seq1_truth_norm, model1, 1))
my_thread2 = threading.Thread(target=trainLSTM, args=(history_list, X2, y2, epoch, 0, in_seq2_test_norm, in_seq2_truth_norm, model2, 2))
my_thread3 = threading.Thread(target=trainLSTM, args=(history_list, X3, y3, epoch, 0, in_seq3_test_norm, in_seq3_truth_norm, model3, 3))
my_thread4 = threading.Thread(target=trainLSTM, args=(history_list, X4, y4, epoch, 0, in_seq4_test_norm, in_seq4_truth_norm, model4, 4))
my_thread5 = threading.Thread(target=trainLSTM, args=(history_list, X5, y5, epoch, 0, in_seq5_test_norm, in_seq5_truth_norm, model5, 5))
my_thread6 = threading.Thread(target=trainLSTM, args=(history_list, X6, y6, epoch, 0, in_seq6_test_norm, in_seq6_truth_norm, model6, 6))
my_thread7 = threading.Thread(target=trainLSTM, args=(history_list, X7, y7, epoch, 0, in_seq7_test_norm, in_seq7_truth_norm, model7, 7))


"""
Plot val_loss and train_loss:
    theta = 0
    #Plot de la validación:
    fig, ax = pyplot.subplots(figsize=(8, 6))
    pyplot.plot(history_list[theta].history['loss'])
    pyplot.plot(history_list[theta].history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.ylim((0, 1))
    pyplot.show()
"""


my_thread0.start()
my_thread1.start()
my_thread2.start()
my_thread3.start()
my_thread4.start()
my_thread5.start()
my_thread6.start()
my_thread7.start()

"""
#Esperamos a que todos finalicen:
"""

my_thread0.join()
my_thread1.join()
my_thread2.join()
my_thread3.join()
my_thread4.join()
my_thread5.join()
my_thread6.join()
my_thread7.join()

#Predicciones:
coef0 = predict(model=model0, in_seq=in_seq0, in_seq_test_norm=in_seq0_test_norm, in_seq_truth=in_seq0_truth, recurrent_forecast=recurrent_forecast, normalization=normalization, minimo=minimo0, maximo=maximo0, inicio_val=inicio_val0, final_val=final_val0, timesteps_past=timesteps_past, n_subseqs=n_subseqs0, n_steps=n_steps0, timesteps_future_to_predict=timesteps_future_to_predict, timesteps_future=timesteps_future, theta_series_v2=theta_series_v2, theta=0, CNN=CNN)
coef1 = predict(model=model1, in_seq=in_seq1, in_seq_test_norm=in_seq1_test_norm, in_seq_truth=in_seq1_truth, recurrent_forecast=recurrent_forecast, normalization=normalization, minimo=minimo1, maximo=maximo1, inicio_val=inicio_val0, final_val=final_val0, timesteps_past=timesteps_past, n_subseqs=n_subseqs0, n_steps=n_steps0, timesteps_future_to_predict=timesteps_future_to_predict, timesteps_future=timesteps_future, theta_series_v2=theta_series_v2, theta=1, CNN=CNN)
coef2 = predict(model=model2, in_seq=in_seq2, in_seq_test_norm=in_seq2_test_norm, in_seq_truth=in_seq2_truth, recurrent_forecast=recurrent_forecast, normalization=normalization, minimo=minimo2, maximo=maximo2, inicio_val=inicio_val0, final_val=final_val0, timesteps_past=timesteps_past, n_subseqs=n_subseqs0, n_steps=n_steps0, timesteps_future_to_predict=timesteps_future_to_predict, timesteps_future=timesteps_future, theta_series_v2=theta_series_v2, theta=2, CNN=CNN)
coef3 = predict(model=model3, in_seq=in_seq3, in_seq_test_norm=in_seq3_test_norm, in_seq_truth=in_seq3_truth, recurrent_forecast=recurrent_forecast, normalization=normalization, minimo=minimo3, maximo=maximo3, inicio_val=inicio_val0, final_val=final_val0, timesteps_past=timesteps_past, n_subseqs=n_subseqs0, n_steps=n_steps0, timesteps_future_to_predict=timesteps_future_to_predict, timesteps_future=timesteps_future, theta_series_v2=theta_series_v2, theta=3, CNN=CNN)
coef4 = predict(model=model4, in_seq=in_seq4, in_seq_test_norm=in_seq4_test_norm, in_seq_truth=in_seq4_truth, recurrent_forecast=recurrent_forecast, normalization=normalization, minimo=minimo4, maximo=maximo4, inicio_val=inicio_val0, final_val=final_val0, timesteps_past=timesteps_past, n_subseqs=n_subseqs0, n_steps=n_steps0, timesteps_future_to_predict=timesteps_future_to_predict, timesteps_future=timesteps_future, theta_series_v2=theta_series_v2, theta=4, CNN=CNN)
coef5 = predict(model=model5, in_seq=in_seq5, in_seq_test_norm=in_seq5_test_norm, in_seq_truth=in_seq5_truth, recurrent_forecast=recurrent_forecast, normalization=normalization, minimo=minimo5, maximo=maximo5, inicio_val=inicio_val0, final_val=final_val0, timesteps_past=timesteps_past, n_subseqs=n_subseqs0, n_steps=n_steps0, timesteps_future_to_predict=timesteps_future_to_predict, timesteps_future=timesteps_future, theta_series_v2=theta_series_v2, theta=5, CNN=CNN)
coef6 = predict(model=model6, in_seq=in_seq6, in_seq_test_norm=in_seq6_test_norm, in_seq_truth=in_seq6_truth, recurrent_forecast=recurrent_forecast, normalization=normalization, minimo=minimo6, maximo=maximo6, inicio_val=inicio_val0, final_val=final_val0, timesteps_past=timesteps_past, n_subseqs=n_subseqs0, n_steps=n_steps0, timesteps_future_to_predict=timesteps_future_to_predict, timesteps_future=timesteps_future, theta_series_v2=theta_series_v2, theta=6, CNN=CNN)
coef7 = predict(model=model7, in_seq=in_seq7, in_seq_test_norm=in_seq7_test_norm, in_seq_truth=in_seq7_truth, recurrent_forecast=recurrent_forecast, normalization=normalization, minimo=minimo7, maximo=maximo7, inicio_val=inicio_val0, final_val=final_val0, timesteps_past=timesteps_past, n_subseqs=n_subseqs0, n_steps=n_steps0, timesteps_future_to_predict=timesteps_future_to_predict, timesteps_future=timesteps_future, theta_series_v2=theta_series_v2, theta=7, CNN=CNN)

#Proyectar en PCA:
coefs_predichos_pca = model.transform(np.array([coef0, coef1, coef2, coef3, coef4, coef5, coef6, coef7]).reshape(1, n+1))

































#Testeo de la predicción en la serie temporal: Aquí se asume que el último valor del array de predicción es el coeficiente buscado en cada caso
t = np.linspace(-(Tsventana-1), 0, Tsventana) + math.ceil(scope/2)
t_step = 1/(2*t[-1])
t = t*t_step

#pol = coef0 + coef1*t + coef2*pow(t,2) + coef3*pow(t,3) + coef4*pow(t,4) + coef5*pow(t,5) + coef6*pow(t,6)
pol = coef0 + coef1*t + coef2*pow(t,2) + coef3*pow(t,3) + coef4*pow(t,4) + coef5*pow(t,5) + coef6*pow(t,6) + coef7*pow(t,7)
fig, ax = pyplot.subplots(figsize=(8, 6)) #New figure
ax.plot(t, pol);

#True pol:
coef0_truth = in_seq0_truth[-1]
coef1_truth = in_seq1_truth[-1]
coef2_truth = in_seq2_truth[-1]
coef3_truth = in_seq3_truth[-1]
coef4_truth = in_seq4_truth[-1]
coef5_truth = in_seq5_truth[-1]
coef6_truth = in_seq6_truth[-1]
coef7_truth = in_seq7_truth[-1]



print('Real vs Predicted')
print(str(round(coef0_truth, 2)) + ' --- ' + str(round(coef0, 2))+'\n')
print(str(round(coef1_truth, 2)) + ' ---' + str(round(coef1, 2))+'\n')
print(str(round(coef2_truth, 2)) + ' --- ' + str(round(coef2, 2))+'\n')
print(str(round(coef3_truth, 2)) + ' --- ' + str(round(coef3, 2))+'\n')
print(str(round(coef4_truth, 2)) + ' --- ' + str(round(coef4, 2))+'\n')
print(str(round(coef5_truth, 2)) + ' --- ' + str(round(coef5, 2))+'\n')
print(str(round(coef6_truth, 2)) + ' --- ' + str(round(coef6, 2))+'\n')
print(str(round(coef7_truth, 2)) + ' --- ' + str(round(coef7, 2))+'\n')

#pol_truth = coef0_truth + coef1_truth*t + coef2_truth*pow(t,2) + coef3_truth*pow(t,3) + coef4_truth*pow(t,4) + coef5_truth*pow(t,5) + coef6_truth*pow(t,6)
pol_truth = coef0_truth + coef1_truth*t + coef2_truth*pow(t,2) + coef3_truth*pow(t,3) + coef4_truth*pow(t,4) + coef5_truth*pow(t,5) + coef6_truth*pow(t,6) + coef7_truth*pow(t,7)

ax.plot(t, pol_truth);
pyplot.legend(['prediction', 'real'], loc='upper right')
pyplot.vlines(t[-scope], 0, 1.05*max(Window_Selected), linestyles ="dashed", colors="r")
ax.set_xlim(t[0], t[-1])
ax.plot(t, Window_Selected, linewidth=0.3);
pyplot.ylim((0.95*min(Window_Selected), 1.05*max(Window_Selected)))


#Polinomio total:
#pol_parte_conocida = pol[0:len(pol)-scope]-coef0
Window_Selected_parte_conocida = Window_Selected[0:len(pol)-scope]

#x = np.array(range(len(Window_Selected_parte_conocida)))+1
x = t[:len(Window_Selected_parte_conocida)]
coeffs = np.polyfit(x, Window_Selected_parte_conocida, n)
pol_parte_conocida = np.polyval(coeffs, x)

pol_parte_desconocida = pol[-scope:]
pol_total = np.concatenate((pol_parte_conocida, pol_parte_desconocida))
#ax.plot(t, pol_total);

print('Comparación entre polinomio de la ventana total (superparte) y de la ventana conocida (subparte):\n')
print('Superparte - Subparte\n')
print(str(coef0_truth) + ' - ' + str(coeffs[-1])+'\n')
print(str(coef1_truth) + ' - ' + str(coeffs[-2])+'\n')
print(str(coef2_truth) + ' - ' + str(coeffs[-3])+'\n')
print(str(coef3_truth) + ' - ' + str(coeffs[-4])+'\n')
print(str(coef4_truth) + ' - ' + str(coeffs[-5])+'\n')
print(str(coef5_truth) + ' - ' + str(coeffs[-6])+'\n')
print(str(coef6_truth) + ' - ' + str(coeffs[-7])+'\n')
print(str(coef7_truth) + ' - ' + str(coeffs[-8])+'\n')
"""
fig, ax = pyplot.subplots(figsize=(8, 6))
ax.plot(x, pol_parte_conocida);
ax.plot(x, Window_Selected_parte_conocida);
"""

coefs_reales = [coef0_truth, coef1_truth, coef2_truth, coef3_truth, coef4_truth, coef5_truth, coef6_truth, coef7_truth]
coefs_predicted = [coef0, coef1, coef2, coef3, coef4, coef5, coef6, coef7]
np.savetxt('outputs/polParteConocida'+str(tiempo_final)+str(recurrent_forecast)+'.txt', pol_parte_conocida, delimiter=',');
np.savetxt('outputs/polParteDesconocida'+str(tiempo_final)+str(recurrent_forecast)+'.txt', pol_parte_desconocida, delimiter=',');
np.savetxt('outputs/WindowTest'+str(tiempo_final)+str(recurrent_forecast)+'.txt', Window_Selected, delimiter=',');
np.savetxt('outputs/polTruth'+str(tiempo_final)+str(recurrent_forecast)+'.txt', pol_truth, delimiter=',');
np.savetxt('outputs/polPredicted'+str(tiempo_final)+str(recurrent_forecast)+'.txt', pol, delimiter=',');
np.savetxt('outputs/theta_truth'+str(tiempo_final)+str(recurrent_forecast)+'.txt', coefs_reales, delimiter=',');
np.savetxt('outputs/theta_predicted'+str(tiempo_final)+str(recurrent_forecast)+'.txt', coefs_predicted, delimiter=',');
#Plot vs real time range:
#Close all figures: pyplot.close('all')
'''
tref = 1465775999;
Tsventana = 30*60;
final = 24601+tref+scope;
inicio = final-Tsventana;
time_range_epoch = np.linspace(inicio+1, final, final-inicio)
tiempoventana = []
for e in time_range_epoch:
    tiempoventana.append(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(e-7200)))

fig, ax = pyplot.subplots(figsize=(8, 6))
ax.plot(tiempoventana, pol);
ax.plot(tiempoventana, pol_truth);
pyplot.legend(['Prediction', 'Real'], loc='upper right')
pyplot.vlines(tiempoventana[-scope], 0, 1.05*max(Window_Selected), linestyles ="dashed", colors="r")
ax.set_xlim(tiempoventana[0], tiempoventana[-1])
ax.plot(tiempoventana, Window_Selected, linewidth=0.3);
pyplot.ylim((0.95*min(Window_Selected), 1.05*max(Window_Selected)))
'''
coefs_truth_pca = model.transform(np.array([coef0_truth, coef1_truth, coef2_truth, coef3_truth, coef4_truth, coef5_truth, coef6_truth, coef7_truth]).reshape(1, n+1))
fig, ax = pyplot.subplots(figsize=(8, 6)) #New figure
pyplot.scatter(X_data_norm[:, 0], X_data_norm[:, 1], alpha=0.2)
pyplot.scatter(coefs_predichos_pca[:, 0], coefs_predichos_pca[:, 1], alpha=1, color="red")
pyplot.scatter(coefs_truth_pca[:, 0], coefs_truth_pca[:, 1], alpha=1)


#Plot real pol. vs previous pols:
fig, ax = pyplot.subplots(figsize=(8, 6)) #New figure
ax.plot(t, pol_truth);
coefs_reales = [coef0_truth, coef1_truth, coef2_truth, coef3_truth, coef4_truth, coef5_truth, coef6_truth, coef7_truth]
np.savetxt('pol_'+str(0)+'.txt', coefs_reales, delimiter=',');
np.savetxt('pol_'+str(10)+'.txt', coefs_predicted, delimiter=',');
for i in range(1):
    i=i+1
    coef0_previous1 = in_seq0[-i]
    coef1_previous1 = in_seq1[-i]
    coef2_previous1 = in_seq2[-i]
    coef3_previous1 = in_seq3[-i]
    coef4_previous1 = in_seq4[-i]
    coef5_previous1 = in_seq5[-i]
    coef6_previous1 = in_seq6[-i]
    coef7_previous1 = in_seq7[-i]
    
    coefs_predicted = [coef0_previous1, coef1_previous1, coef2_previous1, coef3_previous1, coef4_previous1, coef5_previous1, coef6_previous1, coef7_previous1]
    np.savetxt('pol_'+str(i)+'.txt', coefs_predicted, delimiter=',');
    pol_previous = coef0_previous1 + coef1_previous1*t + coef2_previous1*pow(t,2) + coef3_previous1*pow(t,3) + coef4_previous1*pow(t,4) + coef5_previous1*pow(t,5) + coef6_previous1*pow(t,6) + coef7_previous1*pow(t,7)
    
    ax.plot(t-i, pol_previous);

window_selected_index = round((tiempo_final+scope)/diezmado)
Window_Selected2 = time_serie[window_selected_index*diezmado-i*scope:window_selected_index*diezmado+Tsventana]
np.savetxt('windowSelectedExpandedParaMemoria.txt', Window_Selected2, delimiter=',');