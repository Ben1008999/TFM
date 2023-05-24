"""
Created on Wed Nov  2 20:54:46 2022
@author: Benjamín Martín Gómez
"""
#Enlaces utilizados para este codigo:
#https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
#https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
#https://datascience.stackexchange.com/questions/43191/validation-loss-is-not-decreasing
#https://stats.stackexchange.com/questions/425610/why-massive-random-spikes-of-validation-loss
#https://datascience.stackexchange.com/questions/43191/validation-loss-is-not-decreasing
#https://stackoverflow.com/questions/61287322/validation-loss-sometimes-spiking En este enlace se explica que un tamaño de batch más pequeño produce spikes en las pérdidas de validación
import math
import threading
import time
import numpy as np
import pandas as pd
from numpy import array
from numpy import hstack
from matplotlib import pyplot

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
#Mi(s) libreria(s):
import dataNormalization as dn

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
        ax.plot(real)
        ax.plot(prediction)
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
        #Función que obtiene una serie temporal de los parámetros theta (ya diezmada) y diferentes parámetros del entrenamiento y crea los datos de entrenamiento dimensionados de forma apropiada para ser insertados en la LSTM
        #En concreto, devuelve:
            # X: Matriz de características
            # y: Matriz de etiquetas
            # in_seq: Secuencia (vector de datos) que representa los datos usados para el entrenamiento de la red. De esta secuencia se crean los patrones pasado - futuro y además contiene todo:
                #in_seq contiene tanto los datos de entrenamiento como los datos de test y de validación (Ground Truth) 
            # in_seq_truth: Secuencia Ground Truth que se refiere a la secuencia predicha real
            # in_seq_test_norm: Secuencia de test que se introducirá a la red LSTM normalizada (esta es la secuencia que realmente se usa como test en la red LSTM) y con la que la red obtendrá una salida de predicción
            # in_seq_truth_norm: Secuencia Ground Truth que se refiere a la secuencia predicha real normalizada
            # model: objeto con la información del modelo
            # inicio_val: Instante de tiempo inicial de la validación (incluido)
            # final_val: Instante de tiempo final de la validación (incluido)
            # timesteps_past: puntos pasados (memoria de la red LSTM)
            # minimo: parámetro de normalización 1
            # maximo: parámetro de normalización 1
            # n_steps: número de puntos de la subsecuencia (en caso de usarla)
            # n_subseqs: Número de subsecuencias (en caso de usarlas)

        kernel_size = round(time_neighbour_points/diezmado)
        T_train = round(T_train/diezmado) #Tamaño de ventana de tiempo de entrenamiento [s]
        timesteps_past = round(time_past/diezmado)
        n_steps = round(time_subseqs/diezmado) #Número de puntos de la subsecuencia (en caso de usarla)

        rest = divmod(timesteps_past, n_steps)
        if rest[1] is not 0:
            n_steps = get_closer_subseq_nsteps(timesteps_past, n_steps)
        n_subseqs = int(timesteps_past/n_steps) #Número de subsecuencias

        #Train:
        inicio_train = final - T_train + 1
        
        #Por ejemplo, tomamos información de las 2 semanas anteriores y de la semana actual:
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

        if CNN == 1: #Para CNN + LSTM, se debe dimensionar la matriz de entrenamiento de una forma concreta:
            X = X.reshape((X.shape[0], n_subseqs, n_steps, n_features))
        if CNN == 0: #Para LSTM simple:
            X = X.reshape((X.shape[0], X.shape[1], n_features)); y = y.reshape((y.shape[0], y.shape[1], n_features))
        
        #Test:
        inicio_test = final - timesteps_past + 1
        in_seq_test = theta_series_v2[semana][theta][inicio_test:final+1] #Serie temporal theta0
        if normalization == 0:
            in_seq_test_norm = dn.normalizeData_MinMax_using_scalerData(in_seq_test, minimo, maximo)
        if normalization == 1:
            in_seq_test_norm = dn.normalizeData_tanh_using_scalerData(in_seq_test, minimo, maximo)
        if normalization == 2:
            in_seq_test_norm = dn.normalizeData_zscore_using_scalerData(in_seq_test, minimo, maximo)

        #Para CNN + LSTM:
        if CNN == 1:
            in_seq_test_norm = in_seq_test_norm.reshape((1, n_subseqs, n_steps, n_features))
        #Para LSTM simple:
        if CNN == 0:
            in_seq_test_norm = in_seq_test_norm.reshape((1, timesteps_past, n_features))

        #Truth (validation)
        inicio_val = (final + 1); final_val = inicio_val+timesteps_future-1
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

        #CNN + LSTM:
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
    #Función que recibe una lista de listas y, por cada lista, las invierte:
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
    history = model.fit(X, y, epochs=epochs, verbose=verbose, validation_data=(test_norm, truth_norm), callbacks=[trainingStopCallback]) #callbacks=[trainingStopCallback]
    list[ide] = history




#--------------------------------------------------Código principal---------------------------------------------------------------------
if __name__=="__main__":
    pyplot.close('all') #Se cierran las posibles figuras que hubiera antes de ejecutar este código

    #Variables de entrada:--------------------------------------------------------------------------------------------------------------
    Tsventana = 30*60 #Tamaño de ventana que se usó para sacar los coeficientes y parámetros alpha-stable (poner mismo valor que en TrendDynamics)
    n = 7 #Grado de la regresión polinómica que se usó cuando se obtuvieron estos parámetros theta (poner mismo valor que en TrendDynamics)
    timesteps_future = 1 #Número de puntos futuros usados para la predicción
    timesteps_future_recurrent = 1 #Número de tiempos usados para la predicción recurrente
    diezmado = 180 #[s] Período de muestreo de las series temporales de los coeficientes theta/alpha
    recurrent_forecast = 0 #0: no se desea predicción recurrente. 1: se desea predicción recurrente (se adivina un punto y se usa para la siguiente predicción y así hasta completar los multistep puntos)
    normalization = 2 #0: MinMax. 1: tanh. 2: zscore
    CNN = 0 #1: CNN + LSTM. 0: LSTM simple
    filename_thetaParams = "TP30_7.txt" #Fichero con los parámetros theta de entrenamiento obtenidos de TrendDynamics.m
    filename_network_traffic_series = 'All_series.txt' #Fichero con todas las series temporales (matriz agregado de TrendDynamics.m)
    semana = 9 #Indexada desde el 0 incluido. Semana de test. A continuación, se indica el índice de cada semana (de este repositorio):
    tiempo_final = 27849 #Instante de tiempo final [s] (de las series theta_i) del que se conocen datos (en segundos, desde las 00:00:00 del lunes)
    #Por ejemplo, si se pone como tiempo_final = 86400, entonces este código va a coger toda la información de los parámetros theta/alpha que va desde el lunes 00:00:00 hasta el martes 00:00:00 incluidos ambos
    #Cabe destacar que el tiempo_final no es un valor libre, sino limitado al diezmado. En un ejemplo simple, si tiempo_final es 10s y el diezmado es de 3s, entonces no es posible tomar la muestra 10 de las series temporales
    #de theta_i, porque 10 no es múltiplo de 3; solo se podrían tomar los segundos 0, 3, 6, 9... Por ello, en este código hay una etapa encargada de redondear y tomar la muestra
    #más cercana a la muestra indicada en "tiempo_final" (en el ejemplo, sería la 9, por ser la más cercana a tiempo_final = 10)
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
    #Parámetros de la red LSTM y del entrenamiento:
    #Estos parámetros se definen para una red LSTM concreta (la de theta0) y en este Script se usan los mismos parámetros para el resto de las redes (del resto de parámetros). Sin embargo, cada red LSTM se podría configurar independientemente del resto. Para ello,
    #más abajo, en la sección de modelos de entrenamiento, en create_model y en train_LSTM se pueden pasar distintos argumentos.
    time_past0 = (360/diezmado)*diezmado #Tiempo pasado [s] usado para el aprendizaje de patrones pasado-futuro (memoria de la red LSTM)
    time_subseqs0 = time_past0 #En el caso de la red LSTM convolucional, toma subsecuencias con duración time_subseqs0. No es estrictamente necesario ni siquiera para una red LSTM, así que se puede dejar con el mismo valor que time_past
    T_train0 = 3000*diezmado #Tiempo pasado usado para el entrenamiento (por ejemplo, la red aprende cogiendo datos de los 3 días anteriores, 4 días, 5 días...)
    time_neighbour_points0 = 1*diezmado; #Solo para CNN + LSTM:Tiempo [s] de análisis que se usa en la capa convolucional para evaluar muestras vecinas por cada subsecuencia
    epoch = 100 #Número de épocas usadas para el entrenamiento
    #-----------------------------------------------------------------------------------------------------------------------------------

    #Si por ejemplo diezmado = 1s y timesteps_future = 1, queremos evaluar la próxima ventana que está a diezmado*timesteps_future de distancia temporal (en este ejemplo, a 1*1s de distancia)
    #Si diezmado = 60s y timesteps_future = 1, entonces queremos hacer una predicción unistep (porque timestep future es 1) usando un diezmado cada 60s, así que el punto que se va a predecir es la próxima ventana que dista 60*1 segs de la última ventana conocida
    #Si diezmado = 30s y timesteps_future = 2, es el mismo caso que antes, solo que ahora lo vamos a hacer como una predicción multistep: muestreamos la señal theta cada 30s y predecimos los próximos dos puntos, que en realidad es predecir los coeficientes en la ventana que dista 30*2 = 60s de la última ventana conocida de la que se tienen datos
    #Por tanto, se define el alcance como el producto diezmado*timesteps_future:
    scope = timesteps_future*diezmado
    print('Model forecast coefficients for the next '+str(scope)+' seconds\n')
    #Lectura de ficheros:
    thetaParams = np.loadtxt(filename_thetaParams, delimiter=',')
    all_series = np.loadtxt(filename_network_traffic_series, delimiter=',')
    #Inicialización de la matriz de datos que contiene las series temporales de los coeficientes del polinomio:
    n_coefs = n+1
    n_series = int(thetaParams.shape[1]/n_coefs)
    n_ventanas = math.ceil(thetaParams.shape[0]/diezmado)
    theta_series = np.zeros((n_coefs, n_series, n_ventanas))
    #theta_series[0] = paquete con las series temporales de theta 0 ordenadas de arriba a abajo en orden semanal. Así, la primera fila del paquete theta_series[0] es la serie temporal de theta0 de la última semana
        #theta_series[0][0] = serie theta 0 más antigua (marzo semana 3)
        #theta_series[0][1] = serie theta 0 de la semana siguiente a la más antigua
        #...
    #theta_series[1] = paquete con las series temporales de theta 1...
    theta_series_v2 = np.zeros((n_series, n_coefs, n_ventanas))
    #También se almacenan las series temporales de la siguiente manera para facilitar la concatenación por semanas posterior:
    #theta_series_v2[0] = Parámetros theta de la serie temporal de marzo week3
        #theta_series_v2[0][0] = Parámetro theta0 de la serie temporal de marzo week3
        #theta_series_v2[0][1] = Parámetro theta1 de la serie temporal de marzo week3
        #...
    #theta_series_v2[1] = Parámetros theta de la serie temporal de marzo week4
    #theta_series_v2[2] = Parámetros theta de la serie temporal de marzo week5
    final = round(tiempo_final/diezmado)
    tiempo_final = final*diezmado

    hora_final = tiempo_final/3600
    minuto_final = (hora_final-int(hora_final))*60
    segundo_final = (minuto_final-int(minuto_final))*60

    print('Última ventana conocida: Lunes [00:00:00] + ['+str(int(hora_final))+'h, '+str(int(minuto_final))+'min, '+str(int(segundo_final))+'s]\n')
    #------------------------------------------------------------------------------
    #Diezmado de las series temporales de los parámetros theta:
    for c in range(1, n_coefs+1): #Por cada coeficiente (hay n+1 coeficientes. Recorremos de 1 a n_coefs. Ponemos +1 porque en python el último valor no cuenta)
        for serie in range(1, n_series+1): #Por cada serie
            #Tomamos los valores cada "diezmado" segundos:
            theta_evolution = thetaParams[::diezmado, (serie-1)*n_coefs+c-1]
            theta_series[c-1][serie-1] = theta_evolution
    #Para el caso de theta_series_v2:
    for serie in range(1, n_series+1):
        for c in range(1, n_coefs+1):
            theta_evolution = thetaParams[::diezmado, (serie-1)*n_coefs+c-1]
            theta_series_v2[serie-1][c-1] = theta_evolution
    #------------------------------------------------------------------------------
    #Prueba de entrenamiento y predicción de parámetros theta:
    time_serie = all_series[semana]
    n_features = 1 #Número de series temporales por cada LSTM (1 por cada LSTM)
    history_list = [0]*(n+1) #Inicialización de una lista de objetos que contendrá información sobre el entrenamiento realizado. Cada elemento de la lista es el resultado de un entrenamiento de una serie temporal (de un coeficiente theta_i)
    timesteps_future_to_predict = 0 #Esta variable se refiere al número de puntos futuros que predecir en el caso de la predicción recurrente (de 1 en 1, de 2 en 2, de 3 en 3 etc).
    #Esta variable es por defecto igual a 0 (predicción no recurrente, sino multistep: se obtiene toda la predicción de una sola vez), porque no se usa en ese caso.

    if recurrent_forecast == 1: #Pero en caso de ser una predicción recurrente:
        timesteps_future_to_predict = timesteps_future #Timesteps futuros que puso el usuario en su momento (arriba)
        timesteps_future = timesteps_future_recurrent
        resultdivmod = divmod(timesteps_future_to_predict, timesteps_future)
        #Si el usuario quiere predecir los próximos 10 segundos de 3 en 3, no es posible: debe ser divisible
        if resultdivmod[1] is not 0:
            print('WARNING: El usuario ha introducido un valor de timesteps_future para la predicción recurrente que no es divisible con el número de timesteps futuros que se quiere predecir. Se van a predecir algunos puntos más de los indicados\n')

    window_selected_index = round((tiempo_final+scope)/diezmado) #Índice de la ventana de tráfico de test
    Window_Selected = time_serie[window_selected_index*diezmado:window_selected_index*diezmado+Tsventana] #Ventana de tráfico de test en bps/pps...



    #THETA0:--------------------------------------------------------------------------------------------------
    X0, y0, in_seq0, in_seq0_truth, in_seq0_test_norm, in_seq0_truth_norm, model0, inicio_val0, final_val0, timesteps_past, minimo0, maximo0, n_steps0, n_subseqs0 = create_model(theta=0, time_past=time_past0, time_subseqs=time_subseqs0, T_train=T_train0, time_neighbour_points=time_neighbour_points0, normalization=normalization, CNN=CNN, theta_series_v2=theta_series_v2)
    X1, y1, in_seq1, in_seq1_truth, in_seq1_test_norm, in_seq1_truth_norm, model1, inicio_val1, final_val1, timesteps_past, minimo1, maximo1, n_steps1, n_subseqs1 = create_model(theta=1, time_past=time_past0, time_subseqs=time_subseqs0, T_train=T_train0, time_neighbour_points=time_neighbour_points0, normalization=normalization, CNN=CNN, theta_series_v2=theta_series_v2)
    X2, y2, in_seq2, in_seq2_truth, in_seq2_test_norm, in_seq2_truth_norm, model2, inicio_val2, final_val2, timesteps_past, minimo2, maximo2, n_steps2, n_subseqs2 = create_model(theta=2, time_past=time_past0, time_subseqs=time_subseqs0, T_train=T_train0, time_neighbour_points=time_neighbour_points0, normalization=normalization, CNN=CNN, theta_series_v2=theta_series_v2)
    X3, y3, in_seq3, in_seq3_truth, in_seq3_test_norm, in_seq3_truth_norm, model3, inicio_val3, final_val3, timesteps_past, minimo3, maximo3, n_steps3, n_subseqs3 = create_model(theta=3, time_past=time_past0, time_subseqs=time_subseqs0, T_train=T_train0, time_neighbour_points=time_neighbour_points0, normalization=normalization, CNN=CNN, theta_series_v2=theta_series_v2)
    X4, y4, in_seq4, in_seq4_truth, in_seq4_test_norm, in_seq4_truth_norm, model4, inicio_val4, final_val4, timesteps_past, minimo4, maximo4, n_steps4, n_subseqs4 = create_model(theta=4, time_past=time_past0, time_subseqs=time_subseqs0, T_train=T_train0, time_neighbour_points=time_neighbour_points0, normalization=normalization, CNN=CNN, theta_series_v2=theta_series_v2)
    X5, y5, in_seq5, in_seq5_truth, in_seq5_test_norm, in_seq5_truth_norm, model5, inicio_val5, final_val5, timesteps_past, minimo5, maximo5, n_steps5, n_subseqs5 = create_model(theta=5, time_past=time_past0, time_subseqs=time_subseqs0, T_train=T_train0, time_neighbour_points=time_neighbour_points0, normalization=normalization, CNN=CNN, theta_series_v2=theta_series_v2)
    X6, y6, in_seq6, in_seq6_truth, in_seq6_test_norm, in_seq6_truth_norm, model6, inicio_val6, final_val6, timesteps_past, minimo6, maximo6, n_steps6, n_subseqs6 = create_model(theta=6, time_past=time_past0, time_subseqs=time_subseqs0, T_train=T_train0, time_neighbour_points=time_neighbour_points0, normalization=normalization, CNN=CNN, theta_series_v2=theta_series_v2)
    X7, y7, in_seq7, in_seq7_truth, in_seq7_test_norm, in_seq7_truth_norm, model7, inicio_val7, final_val7, timesteps_past, minimo7, maximo7, n_steps7, n_subseqs7 = create_model(theta=7, time_past=time_past0, time_subseqs=time_subseqs0, T_train=T_train0, time_neighbour_points=time_neighbour_points0, normalization=normalization, CNN=CNN, theta_series_v2=theta_series_v2)

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



#Testeo de la predicción en la serie temporal: Aquí se asume que el último valor del array de predicción es el coeficiente buscado en cada caso
t = np.linspace(-(Tsventana-1), 0, Tsventana) + math.ceil(scope/2)
t_step = 1/(2*t[-1])
t = t*t_step

pol = coef0 + coef1*t + coef2*pow(t,2) + coef3*pow(t,3) + coef4*pow(t,4) + coef5*pow(t,5) + coef6*pow(t,6) + coef7*pow(t,7)
fig, ax = pyplot.subplots(figsize=(8, 6)) #New figure
ax.plot(t, pol)

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

ax.plot(t, pol_truth)
pyplot.legend(['prediction', 'real'], loc='upper right')
pyplot.vlines(t[-scope], 0, 1.05*max(Window_Selected), linestyles ="dashed", colors="r")
ax.set_xlim(t[0], t[-1])
ax.plot(t, Window_Selected, linewidth=0.3)
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