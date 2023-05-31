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
matplotlib.use('Agg')

    
def createGUI(filename_network_traffic_series, filename_thetaParams):
    #Paso 0: obtener los datos de cada semana:
    all_series = np.loadtxt(filename_network_traffic_series, delimiter=',')
    
    #Paso 1: preparar layout vacío inicialmente:
    root = tk.Tk()
    root.title("Predictor")
    root.geometry("600x800")
    
    week_was = 0
    #Hacer que el usuario escoja una semana:
    def on_change_WEEKBUTTON():
        nonlocal week_was
        week = int(spinbox_WEEK.get())
        if week != week_was:
            #Update the plot of the week:
            time_serie = all_series[week]
            #Clean plot:
            my_line[0].set_data([], [])
            time_serie = all_series[week]
            #Plot again:
            my_line[0] = ax.plot(time_serie, linewidth=0.1)[0]

            #Pintar plot en la GUI:
            canvas.draw()
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
    fig, ax = plt.subplots()
    plt.xlim(1, len(all_series[0]))
    plt.ylim(1, 150000)
    plt.grid(True, alpha=0.5)
    my_line = ax.plot(time_serie, linewidth=0.1)
    
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.get_tk_widget().pack()
    
    # create the textbox widget and set its textvariable
    #textbox = ttk.Entry(root, textvariable=text_var_WEEKTEXTBOX, width=300)
    #textbox.pack()
    
    
    
        
    # bind the <Return> event to the textbox widget

    #Plot:
    '''
    fig = plt.Figure(figsize=(6, 3), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(time_serie)
    #line = ax.axvline(x=3, color='red')
    #print(ax.name)
    
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()
    
    def update_plot(value):
        # Update the plot based on the slider value
        pass
    
    slider = ttk.Scale(root, orient='horizontal', from_=1, to=len(time_serie), command=update_plot, style="Horizontal.TScale", length=600)
    slider.pack()
    '''
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
createGUI(filename_network_traffic_series, filename_thetaParams)

#El resto del código no se ejecuta hasta que se hayan introducido las variables:
print('Next')