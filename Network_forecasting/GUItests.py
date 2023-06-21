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
import json
from GUIStyleModule import GraphicalUserInterface, GUI_Data
matplotlib.use('Agg')

def getJSONTrendDynamicsData(filenameJSON):
    #Function to get JSON data from JSON filename:
    f = open(filenameJSON)
    dataJSON = json.load(f)
    return dataJSON

def getAllSeriesMatrix(JSONobject):
    #Function to get the matrix with the time series data from JSON object:
    key_list = list(JSONobject.keys())[7:-2]
    all_series = np.zeros((len(key_list), len(JSONobject[key_list[0]])))
    for i in range(len(key_list)):
        time_serie = JSONobject[key_list[i]]
        all_series[i] = time_serie
    return all_series

def getThetaParamsMatrix(JSONobject):
    #Function to get the matrix with theta params of sliding windows from JSON object:
    n = JSONobject["n"]
    Tsventana = JSONobject["Tsventana"]
    thetaParams = np.array(JSONobject["TP"+str(int(Tsventana/60))+'_'+str(n)])
    return thetaParams
    
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

def interpolate(array, Ts):
    result = []
    for i in range(1, len(array)):
        adders = []
        if array[i] == None or array[i-1] == None:
            for x in range(1, Ts):
                adders.append(None)
        else:
            m = (array[i]-array[i-1])/Ts
            for x in range(1, Ts):
                adders.append(array[i-1] + m*x)
        result = result + [array[i-1]] + adders
        
    return result

def map_None_to_nan(element):
    if element is None:
        return np.nan
    return element
def take_not_nan_values(array):
    #Convert all NoneType to np.nan:
    array = list(map(map_None_to_nan, array))
    array = [element for element in array if math.isnan(element) is False]
    return [element for element in array if math.isnan(element) is False]

def take_not_nan_indexes(array):
    index_from = 0
    index_to = 0
    #Convert all NoneType to np.nan:
    array = list(map(map_None_to_nan, array))
    for i in range(len(array)):
        e = array[i]
        if math.isnan(e) is False:
            index_from = i
            break
    for i in range(len(array)-1, -1, -1):
        e = array[i]
        if math.isnan(e) is False:
            index_to = i
            break
    
    return index_from, index_to

def createDefaultGUI(initial_display_week, initial_display_coefficient, all_series, n, Tsventana, granularidad_deteccion, thetaParams):
    #Function to initialize GUI:
    root = tk.Tk()
    root.title("Predictor")
    root.geometry("1080x800")
    
    #Week spinbox:
    #Functionality: when user changes the week, the plot of network traffic week time series must update:
    def on_change_WEEKSPINBOX():
        #Take the chosen current week by user:
        chosen_week = int(GUIobject.spinBoxWeek.get())
        #If selected week is different of previous selected week, update plot of time serie and theta-params:
        if chosen_week != GUIobject.displaying_week_was:
            time_serie = GUIobject.data.all_series[chosen_week]
            GUIobject.time_serie_plot[0].set_data([], [])
            GUIobject.time_serie_plot = ax.plot(time_serie, linewidth=0.1, color='blue')
            GUIobject.time_serie_plot_canvas.draw()
            #Update the theta-params plot:
            theta_index = int(GUIobject.spinBoxTheta.get())
            theta_evolution = GUIobject.data.thetaParams[::GUIobject.samplePeriod, chosen_week*(GUIobject.data.n+1)+theta_index]
            theta_evolution = interpolate(theta_evolution, GUIobject.samplePeriod)
            GUIobject.theta_serie_plot[0].set_data([], [])
            GUIobject.theta_serie_plot = ax_theta.plot(theta_evolution, linewidth=1, color='orange')
            if bool(take_not_nan_values(theta_evolution)):
                y = [0.999*min(take_not_nan_values(theta_evolution)), 1.001*max(take_not_nan_values(theta_evolution))]
                ax_theta.set_ylim(y[0], y[1])
                ax_theta.set_xlim(ax_theta.set_xlim(take_not_nan_indexes(GUIobject.data.thetaParams[::1, chosen_week*(GUIobject.data.n+1)+theta_index])[0], take_not_nan_indexes(GUIobject.data.thetaParams[::1, chosen_week*(GUIobject.data.n+1)+theta_index])[1]))
            else:
                y = [0, 1]
            x = [GUIobject.selectedWindow, GUIobject.selectedWindow]
            GUIobject.theta_serie_line_delimiter_plot[0].set_data([], [])
            GUIobject.theta_serie_line_delimiter_plot = ax_theta.plot(x, y, color='red')
            GUIobject.theta_serie_plot_canvas.draw()
            GUIobject.displaying_week_was = chosen_week
    labelChooseWeek = tk.Label(root, text="Choose here the week:")
    labelChooseWeek.pack()
    spinboxChooseWeek = ttk.Spinbox(root, from_=0, to=all_series.shape[0]-1, increment=1, command=on_change_WEEKSPINBOX)
    spinboxChooseWeek.set(0)
    spinboxChooseWeek.pack()
    
    #Plot de la serie temporal inicial:
    fig, (ax, ax_zoom) = plt.subplots(1, 2, figsize=(10, 5))
    time_serie = all_series[initial_display_week]
    time_serie_plot = ax.plot(time_serie, linewidth=0.1, color='blue')
        #Plot de los límites de la ventana:
    x = [1, 1+Tsventana-1]; y = [0, max(take_not_nan_values(time_serie))]
    window_limits_plot = ax.plot(x, y, color='red')
    ax.set_xlim(1, len(time_serie)); ax.set_ylim(1, 150000); ax.grid(True, alpha=0.5)
        #Plot de la ventana zoom:
    window = time_serie[x[0]-1:x[1]+1]
    time_serie_zoom_plot = ax_zoom.plot(window, linewidth=0.5, color='blue'); ax_zoom.set_xlim(1, len(window))
    #Canvas del plot:
    canvas_netTraffic = FigureCanvasTkAgg(fig, master=root)
    canvas_netTraffic.get_tk_widget().pack()
    
    #DropDown Menu for sampling value:
    #Functionality: when user selects a sampling period, it must update the step of slider and the plot of theta params:
    def on_change_SAMPLINGDROPDOWN(chosen_sample_period):
        GUIobject.windowSliderSelector.config(resolution=chosen_sample_period)
        GUIobject.samplePeriod=chosen_sample_period
        theta_index = int(GUIobject.spinBoxTheta.get())
        week = int(GUIobject.spinBoxWeek.get())
        theta_evolution = GUIobject.data.thetaParams[::chosen_sample_period, week*(GUIobject.data.n+1)+theta_index]
        theta_evolution = interpolate(theta_evolution, chosen_sample_period)
        GUIobject.theta_serie_plot[0].set_data([], [])
        GUIobject.theta_serie_plot = ax_theta.plot(theta_evolution, linewidth=1, color='orange')
        if bool(take_not_nan_values(theta_evolution)):
            y = [0.999*min(take_not_nan_values(theta_evolution)), 1.001*max(take_not_nan_values(theta_evolution))]
            ax_theta.set_ylim(y[0], y[1])
            ax_theta.set_xlim(take_not_nan_indexes(GUIobject.data.thetaParams[::1, week*(GUIobject.data.n+1)+theta_index])[0], take_not_nan_indexes(GUIobject.data.thetaParams[::1, week*(GUIobject.data.n+1)+theta_index])[1])
        else:
            y = [0, 1]
        x = [GUIobject.selectedWindow, GUIobject.selectedWindow]
        GUIobject.theta_serie_line_delimiter_plot[0].set_data([], [])
        GUIobject.theta_serie_line_delimiter_plot = ax_theta.plot(x, y, color='red')
        GUIobject.theta_serie_plot_canvas.draw()
    labelChooseSampling = tk.Label(root, text="Sampling [s]:")
    labelChooseSampling.pack()
    selected_option = tk.StringVar(root)
    selected_option.set("1")
    options = get_divisors(granularidad_deteccion)
    dropdownChooseSampling = tk.OptionMenu(root, selected_option, *options, command=on_change_SAMPLINGDROPDOWN)
    dropdownChooseSampling.pack()
    
    #Window test slider selector:
    #Functionality: when user slides the slider button of test window, it must update the red lines (delimiters of the window), the zoom plot and the last-second line delimiter in theta serie:
    def on_change_WINDOWSLIDER(selectedWindow):
        selectedWindow = int(selectedWindow)
        GUIobject.selectedWindow = selectedWindow
        time_serie = GUIobject.data.all_series[int(GUIobject.spinBoxWeek.get())]
        #Update red lines delimiters:
        x = [selectedWindow, selectedWindow+GUIobject.data.Tsventana-1]
        y = [0, max(take_not_nan_values(time_serie))]
        GUIobject.red_lines_delimiters_plot[0].set_data([], [])
        GUIobject.red_lines_delimiters_plot = ax.plot(x, y, color='red')
        #Update zoom plot:
        window = time_serie[selectedWindow-1:selectedWindow-1+GUIobject.data.Tsventana]
        domain = list(range(selectedWindow,selectedWindow+GUIobject.data.Tsventana))
        GUIobject.window_zoom_plot[0].set_data([], [])
        GUIobject.window_zoom_plot = ax_zoom.plot(window, linewidth=0.5, color='blue')
        ax_zoom.grid(True, alpha=0.5)
        #Draw:
        GUIobject.time_serie_plot_canvas.draw()
        #Get the current theta serie to set the limits of the yline:
        week = int(GUIobject.spinBoxWeek.get())
        sample_period = int(GUIobject.samplePeriod)
        theta_index = int(spinboxChooseThetaSerie.get())
        theta_evolution = GUIobject.data.thetaParams[::sample_period, week*(GUIobject.data.n+1)+theta_index]
        if bool(take_not_nan_values(theta_evolution)):
            y = [0.999*min(take_not_nan_values(theta_evolution)), 1.001*max(take_not_nan_values(theta_evolution))]
            ax_theta.set_ylim(y[0], y[1])
            ax_theta.set_xlim(ax_theta.set_xlim(take_not_nan_indexes(GUIobject.data.thetaParams[::1, week*(GUIobject.data.n+1)+theta_index])[0], take_not_nan_indexes(GUIobject.data.thetaParams[::1, week*(GUIobject.data.n+1)+theta_index])[1]))
        else:
            y = [0, 1]
        x = [GUIobject.selectedWindow, GUIobject.selectedWindow]
        GUIobject.theta_serie_line_delimiter_plot[0].set_data([], [])
        GUIobject.theta_serie_line_delimiter_plot = ax_theta.plot(x, y, color='red')
        #Draw:
        GUIobject.theta_serie_plot_canvas.draw()
        
    labelChooseWindow = tk.Label(root, text="Choose window:")
    labelChooseWindow.pack()
    slider = tk.Scale(root, from_=1, to=len(time_serie)-(Tsventana-1), resolution=1, length=900, orient=tk.HORIZONTAL, command=on_change_WINDOWSLIDER)
    slider.pack()
    
    #Theta serie parameter spinbox:
    #Functionality: when user changes the theta index, it must update the theta series plot:
    def on_change_THETAINDEXSPINBOX():
        theta_index = int(spinboxChooseThetaSerie.get())
        if theta_index != GUIobject.displaying_theta_was:
            week = int(GUIobject.spinBoxWeek.get())
            sample_period = int(GUIobject.samplePeriod)
            theta_evolution = GUIobject.data.thetaParams[::sample_period, week*(GUIobject.data.n+1)+theta_index]
            theta_evolution = interpolate(theta_evolution, GUIobject.samplePeriod)
            GUIobject.theta_serie_plot[0].set_data([], [])
            GUIobject.theta_serie_plot = ax_theta.plot(theta_evolution, linewidth=1, color='orange')
            if bool(take_not_nan_values(theta_evolution)):
                y = [0.999*min(take_not_nan_values(theta_evolution)), 1.001*max(take_not_nan_values(theta_evolution))]
                ax_theta.set_ylim(y[0], y[1])
                ax_theta.set_xlim(ax_theta.set_xlim(take_not_nan_indexes(GUIobject.data.thetaParams[::1, week*(GUIobject.data.n+1)+theta_index])[0], take_not_nan_indexes(GUIobject.data.thetaParams[::1, week*(GUIobject.data.n+1)+theta_index])[1]))
            else:
                y = [0, 1]
            x = [GUIobject.selectedWindow, GUIobject.selectedWindow]
            GUIobject.theta_serie_line_delimiter_plot[0].set_data([], [])
            GUIobject.theta_serie_line_delimiter_plot = ax_theta.plot(x, y, color='red')
            GUIobject.theta_serie_plot_canvas.draw()
            GUIobject.displaying_theta_was = theta_index
        
    labelChooseThetaSerie = tk.Label(root, text="Choose theta parameter time serie:")
    labelChooseThetaSerie.pack()
    spinboxChooseThetaSerie = ttk.Spinbox(root, from_=0, to=n, increment=1, command=on_change_THETAINDEXSPINBOX)
    spinboxChooseThetaSerie.set(0)
    spinboxChooseThetaSerie.pack()
    
    #Plot de la serie temporal theta:
    fig_theta, ax_theta = plt.subplots()
    theta_evolution = thetaParams[::1, initial_display_week*(n+1)+initial_display_coefficient]
    theta_serie_plot = ax_theta.plot(theta_evolution, linewidth=1, color='orange')
    if bool(take_not_nan_values(theta_evolution)):
        y = [0.999*min(take_not_nan_values(theta_evolution)), 1.001*max(take_not_nan_values(theta_evolution))]
        ax_theta.set_ylim(y[0], y[1])
        ax_theta.set_xlim(take_not_nan_indexes(theta_evolution)[0], take_not_nan_indexes(theta_evolution)[1])
    else:
        y = [0, 1]
    x = [1, 1]
    theta_delimiter = ax_theta.plot(x, y, color='red')
    canvas_thetaSerie = FigureCanvasTkAgg(fig_theta, master=root)
    canvas_thetaSerie.get_tk_widget().pack()
    
    GUIdata = GUI_Data(network_time_series = all_series,
                       theta_time_series = thetaParams,
                       Tsventana = Tsventana,
                       n = n)
    
    GUIobject = GraphicalUserInterface(root = root,
                                       data = GUIdata,
                                       spinBoxWeek = spinboxChooseWeek,
                                       time_serie_plot = time_serie_plot,
                                       time_serie_plot_canvas = canvas_netTraffic,
                                       displaying_week_was = initial_display_week,
                                       displaying_theta_was = initial_display_coefficient,
                                       samplingDropDownMenu = dropdownChooseSampling,
                                       samplePeriod = 1,
                                       windowSliderSelector = slider,
                                       selectedWindow = 1,
                                       red_lines_delimiters_plot = window_limits_plot,
                                       window_zoom_plot = time_serie_zoom_plot,
                                       spinBoxTheta = spinboxChooseThetaSerie,
                                       theta_serie_plot = theta_serie_plot,
                                       theta_serie_line_delimiter_plot = theta_delimiter,
                                       theta_serie_plot_canvas = canvas_thetaSerie)
    
    return GUIobject


def createGUI(filenameJSON):
    #Paso 0: obtener los datos de cada semana:
    JSONobject = getJSONTrendDynamicsData(filenameJSON)
    all_series = getAllSeriesMatrix(JSONobject)
    thetaParams = getThetaParamsMatrix(JSONobject)
    n = JSONobject["n"]
    Tsventana = JSONobject["Tsventana"]
    granularidad_deteccion = JSONobject["Scope"]
    #Paso 1: mostrar la interfaz gráfica default:
    GUIobject = createDefaultGUI(initial_display_week=0,
                     initial_display_coefficient=0,
                     all_series=all_series,
                     n=n,
                     Tsventana=Tsventana,
                     granularidad_deteccion=granularidad_deteccion,
                     thetaParams=thetaParams)
    
    
    GUIobject.root.mainloop()
    '''
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
            #ax_theta.set_ylim(0.8*min(theta_evolution), 1.2*max(theta_evolution))
            canvas2.draw() #Para realizar el plot de la serie temporal de theta
            
            week_was = week
        
    label = tk.Label(root, text="Choose here the week:")
    label.pack()
    
    spinbox_WEEK = ttk.Spinbox(root, from_=0, to=all_series.shape[0]-1, increment=1, command=on_change_WEEKBUTTON)
    spinbox_WEEK.set(0)
    spinbox_WEEK.pack()
    
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
    
    canvas = FigureCanvasTkAgg(fig, master=root)
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
        #ax_theta.set_ylim(0.8*min(theta_evolution), 1.2*max(theta_evolution))
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
        #ax_zoom.set_ylim(0.8*min(window), 1.2*max(window))
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
        #ax_theta.set_ylim(0.8*min(theta_evolution), 1.2*max(theta_evolution))
        canvas2.draw() #Para realizar el plot de la serie temporal de theta
    
    spinbox_THETAINDEX = ttk.Spinbox(root, from_=0, to=n, increment=1, command=on_change_THETAINDEX)
    spinbox_THETAINDEX.set(0)
    spinbox_THETAINDEX.pack()
    
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
    '''




filenameJSON = '../Data_extraction/Data_extraction_output/trendDynamicsOutput.json'
createGUI(filenameJSON)

#El resto del código no se ejecuta hasta que se hayan introducido las variables:
print('Next')