# -*- coding: utf-8 -*-
"""
Created on Mon May 29 13:41:19 2023

@author: benja
"""

import tkinter as tk
from tkinter import ttk #Para más opciones de los elementos de la interfaz
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider
import matplotlib
import math
matplotlib.use('Agg')

def get_divisors(x):
    result = []
    for i in range(1, math.floor(x/2)):
        if x % i == 0:
            result.append(i)
    result.append(x)
    return result

def contains_nan(lst):
    for value in lst:
        if math.isnan(value):
            return True
    return False

def createGUI(filename_network_traffic_series, filename_thetaParams, n, granularidad_deteccion, Tsventana):
    #Paso 0: obtener los datos de cada semana:
    all_series = np.loadtxt(filename_network_traffic_series, delimiter=',')
    thetaParams = np.loadtxt(filename_thetaParams, delimiter=',')
    
    #Paso 1: preparar layout vacío inicialmente:
    root = tk.Tk()
    root.title("Predictor")
    root.geometry("1080x800")
    
    week_was = 0
    #Hacer que el usuario escoja una semana:
    def on_change_WEEKBUTTON():
        nonlocal week_was
        nonlocal time_serie
        nonlocal theta_series
        week = int(spinbox_WEEK.get())
        if week != week_was:
            #Update the plot of the week:
            time_serie = all_series[week]
            #Clean plot:
            my_line[0].set_data([], [])
            time_serie = all_series[week]
            #Plot again:
            my_line[0] = ax.plot(time_serie, linewidth=0.1, color='blue')[0]
            canvas.draw()
            
            #También se debe modificar el plot de la serie temporal theta:
            #Tomar el valor de la serie temporal que representar:
            theta_i = int(spinbox_THETAINDEX.get())
            #Tomar el valor de diezmado:
            sample_period = int(selected_option.get())
            theta_evolution = thetaParams[::sample_period, week*n_coefs+theta_i]
            theta_plot[0].set_data([], [])
            theta_plot[0] = ax_theta.plot(theta_evolution, linewidth=1, color='orange')[0]
            if contains_nan(theta_evolution) is False:
                ax_theta.set_ylim(0.8*min(theta_evolution), 1.2*max(theta_evolution))
            canvas2.draw() #Para realizar el plot de la serie temporal de theta
            
            week_was = week
        
    label = tk.Label(root, text="Choose here the week:")
    label.pack()
    
    spinbox_WEEK = ttk.Spinbox(root, from_=0, to=all_series.shape[0]-1, increment=1, command=on_change_WEEKBUTTON)
    spinbox_WEEK.set(0)
    spinbox_WEEK.pack()
    frame = tk.Frame(root)
    frame.pack(pady=20)
    
    #Por defecto, se muestra la serie temporal para la primera semana:
    time_serie = all_series[0]
    fig, (ax, ax_zoom) = plt.subplots(1, 2, figsize=(10, 5))
    my_line = ax.plot(time_serie, linewidth=0.1, color='blue')
    x = [1, 1+Tsventana-1]  # Same x-coordinate for both points
    y = [0, 150000]  # Different y-coordinates to span the desired vertical length
    #Plot the vertical line
    vertical_line = ax.plot(x, y, color='red')
    ax.set_xlim(1, len(all_series[0]))
    ax.set_ylim(1, 150000)
    ax.grid(True, alpha=0.5)
    #También se muestra la primera ventana:
    window = time_serie[x[0]-1:x[1]+1]
    window_plot = ax_zoom.plot(window, linewidth=0.5, color='blue')
    ax_zoom.set_xlim(1, len(window))
    if contains_nan(window) is False:
        ax_zoom.set_ylim(0.8*min(window), 1.2*max(window))
    ax_zoom.grid(True, alpha=0.5)
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.get_tk_widget().pack()
    
    label = tk.Label(root, text="Sampling [s]:")
    label.pack()
    #Escoger el diezmado:
    def handle_selection(selection):
        slider.config(resolution=selection)
        #También se debe modificar el plot de la serie temporal theta:
        #Tomar el valor de la serie temporal que representar:
        theta_i = int(spinbox_THETAINDEX.get())
        week = int(spinbox_WEEK.get())
        #Tomar el valor de diezmado:
        sample_period = int(selected_option.get())
        theta_evolution = thetaParams[::sample_period, week*n_coefs+theta_i]
        theta_plot[0].set_data([], [])
        theta_plot[0] = ax_theta.plot(theta_evolution, linewidth=1, color='orange')[0]
        if contains_nan(theta_evolution) is False:
            ax_theta.set_ylim(0.8*min(theta_evolution), 1.2*max(theta_evolution))
        canvas2.draw() #Para realizar el plot de la serie temporal de theta
    
    options = get_divisors(granularidad_deteccion)
    selected_option = tk.StringVar(root)
    selected_option.set(options[0])
    dropdown = tk.OptionMenu(root, selected_option, *options, command=handle_selection)
    dropdown.pack()
    
    #Sliding button:
    label = tk.Label(root, text="Choose window:")
    label.pack()
    step_increment = int(selected_option.get())
    def on_change_WINDOWSLIDER(selectedWindow):
        selectedWindow = int(selectedWindow)
        vertical_line[0].set_data([], [])
        window_plot[0].set_data([], [])
        x = [selectedWindow, selectedWindow+Tsventana-1]  # Same x-coordinate for both points
        y = [0, 150000]  # Different y-coordinates to span the desired vertical length
        # Plot the vertical line
        vertical_line[0] = ax.plot(x, y, color='red')[0]
        window = time_serie[selectedWindow-1:selectedWindow-1+Tsventana]
        domain = list(range(selectedWindow,selectedWindow+Tsventana))
        window_plot[0] = ax_zoom.plot(domain, window, linewidth=0.5, color='blue')[0]
        ax_zoom.set_xlim(selectedWindow, selectedWindow+Tsventana-1)
        if contains_nan(window) is False:
            ax_zoom.set_ylim(0.8*min(window), 1.2*max(window))
        canvas.draw()
        
    slider = tk.Scale(root, from_=1, to=len(time_serie)-(Tsventana-1), resolution=step_increment, length=500, orient=tk.HORIZONTAL, command=on_change_WINDOWSLIDER)
    slider.pack()
    
    #Escoger serie temporal de parámetros theta:
    label = tk.Label(root, text="Choose theta parameter time serie:")
    label.pack()
    def on_change_THETAINDEX():
        #También se debe modificar el plot de la serie temporal theta:
        #Tomar el valor de la serie temporal que representar:
        theta_i = int(spinbox_THETAINDEX.get())
        week = int(spinbox_WEEK.get())
        #Tomar el valor de diezmado:
        sample_period = int(selected_option.get())
        theta_evolution = thetaParams[::sample_period, week*n_coefs+theta_i]
        theta_plot[0].set_data([], [])
        theta_plot[0] = ax_theta.plot(theta_evolution, linewidth=1, color='orange')[0]
        if contains_nan(theta_evolution) is False: #Las series siempre contienen valores nan
            ax_theta.set_ylim(0.8*min(theta_evolution), 1.2*max(theta_evolution))
        canvas2.draw() #Para realizar el plot de la serie temporal de theta
    
    spinbox_THETAINDEX = ttk.Spinbox(root, from_=0, to=n, increment=1, command=on_change_THETAINDEX)
    spinbox_THETAINDEX.set(0)
    spinbox_THETAINDEX.pack()
    frame = tk.Frame(root)
    frame.pack(pady=20)
    
    #Serie temporal de parámetros theta:
    n_coefs = n+1
    n_series = int(thetaParams.shape[1]/n_coefs)
    n_ventanas = math.ceil(thetaParams.shape[0]) #Por defecto el diezmado es 1
    theta_series = np.zeros((n_coefs, n_series, n_ventanas))
    for c in range(1, n_coefs+1): #Por cada coeficiente (hay n+1 coeficientes. Recorremos de 1 a n_coefs. Ponemos +1 porque en python el último valor no cuenta)
        for serie in range(1, n_series+1): #Por cada serie
            #Tomamos los valores cada "diezmado" segundos:
            theta_evolution = thetaParams[::1, (serie-1)*n_coefs+c-1]
            theta_series[c-1][serie-1] = theta_evolution
    #Por defecto se muestra la serie temporal para theta0 y para la semana 0:
    fig_theta, ax_theta = plt.subplots()
    theta_plot = ax_theta.plot(theta_series[0][0], linewidth=1, color='orange')
    theta_series_aux = list(filter(lambda x: not math.isnan(x), theta_series[0][0]))
    canvas2 = FigureCanvasTkAgg(fig_theta, master=root)
    canvas2.get_tk_widget().pack()
    
    root.mainloop()




filename_network_traffic_series = '../Data_extraction/Data_extraction_output/All_series.txt'
filename_thetaParams = '../Data_extraction/Data_extraction_output/TP30_7.txt'
n = 7


'''
time_serie = all_series[1]

thetaParams = np.loadtxt(filename_thetaParams, delimiter=',')
n_coefs = n+1
n_series = int(thetaParams.shape[1]/n_coefs)
#n_ventanas = math.ceil(thetaParams.shape[0]/diezmado)
#theta_series = np.zeros((n_coefs, n_series, n_ventanas))
'''
n = int(filename_thetaParams.split("_")[-1][:-4])
Tsventana = int(filename_thetaParams.split("_")[-2][-2:])*60
granularidad_deteccion = 180; #[s]
createGUI(filename_network_traffic_series, filename_thetaParams, n, granularidad_deteccion, Tsventana)

#El resto del código no se ejecuta hasta que se hayan introducido las variables:
print('Next')