# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 18:40:37 2023

@author: benja
"""

class GraphicalUserInterface():
    def __init__(self,
                 root = None,
                 #Data:
                 data = None,
                 #Widgets:
                 spinBoxWeek = None,
                 time_serie_plot = None,
                 time_serie_plot_canvas = None,
                 #Memory utilities:
                 displaying_week_was = 0,
                 #Sampling drop down menu:
                 samplingDropDownMenu = None,
                 #Slider:
                 windowSliderSelector = None,
                 #Spinbox for theta serie:
                 spinBoxTheta = None,
                 theta_serie_plot = None,
                 theta_serie_plot_canvas = None
                 ):
        self.root = root
        
        self.data = data
        
        self.spinBoxWeek = spinBoxWeek
        self.displaying_week_was = displaying_week_was
        
        self.time_serie_plot = time_serie_plot
        self.time_serie_plot_canvas = time_serie_plot_canvas
        
        self.samplingDropDownMenu = samplingDropDownMenu
        self.windowSliderSelector = windowSliderSelector
        
        self.spinBoxTheta = spinBoxTheta
        
        self.theta_serie_plot = theta_serie_plot
        self.theta_serie_plot_canvas = theta_serie_plot_canvas
        
        
class GUI_Data():
    def __init__(self,
                 network_time_series=[],
                 theta_time_series=[],
                 n=0):
        self.all_series = network_time_series
        self.thetaParams = theta_time_series
        self.n = n
    