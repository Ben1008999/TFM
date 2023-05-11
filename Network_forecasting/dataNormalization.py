# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 10:21:59 2022

@author: Benjamín Martín Gómez
"""

from numpy import mean, std, tanh
import math
import numpy as np 

def normalizeData_tanh(data):
    #data: array python con los datos
    mu = mean(data)
    sigma = std(data)
    result = (1/2)*(tanh(0.01*((data-mu)/sigma)) + 1)
    return result, mu, sigma

def normalizeData_tanh_using_scalerData(data, mu, sigma):
    #data: array python con los datos
    result = (1/2)*(tanh(0.01*((data-mu)/sigma)) + 1)
    return result

def un_normalizeData_tanh(data_normalized, mu, sigma):
    result = np.zeros((data_normalized.shape[0],))
    for i in range(len(result)):
        result[i] = (math.atanh(2*data_normalized[i]-1)*100)*sigma + mu

    return result




def normalizeData_zscore(data):
    mu = mean(data)
    sigma = std(data)
    result = (data-mu)/sigma
    return result, mu, sigma

def normalizeData_zscore_using_scalerData(data, mu, sigma):
    result = (data-mu)/sigma
    return result

def un_normalizeData_zscore(data, mu, sigma):
    result = data*sigma+mu
    return result




def normalizeData_MinMax(data):
    #data: array python con los datos
    minimo = min(data)
    maximo = max(data)
    #Si se desea que esté entre 0 y 1, descomentar esta línea:
    #result = (data - minimo)/(maximo-minimo)
    
    #Si se desea que esté entre -1 y 1, descomentar estas líneas:
    valor_medio = minimo + (maximo-minimo)/2
    result = (data - valor_medio)/(maximo-valor_medio)
                                   
    return result, minimo, maximo

def normalizeData_MinMax_using_scalerData(data, minimo, maximo):
    #data: array python con los datos
    #Si la normalización fue entre 0 y 1, descomentar esta línea:
    #result = (data - minimo)/(maximo-minimo)
    #Si la normalización fue entre -1 y 1, descomentar estas líneas:
    valor_medio = minimo + (maximo-minimo)/2
    result = (data - valor_medio)/(maximo-valor_medio)
    
    return result

def un_normalizeData_MinMax(data_normalized, minimo, maximo):
    #Si la normalización fue entre 0 y 1, descomentar esta línea:
    #result = data_normalized*(maximo-minimo) + minimo
    #Si la normalización fue entre -1 y 1, descomentar estas líneas:
    valor_medio = minimo + (maximo-minimo)/2
    result = data_normalized*(maximo-valor_medio) + valor_medio
    return result
