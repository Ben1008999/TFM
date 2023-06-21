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
                 displaying_theta_was = 0,
                 #Sampling drop down menu:
                 samplingDropDownMenu = None,
                 samplePeriod = 1,
                 #Slider:
                 windowSliderSelector = None,
                 selectedWindow = 1,
                 red_lines_delimiters_plot = None,
                 window_zoom_plot = None,
                 #Spinbox for theta serie:
                 spinBoxTheta = None,
                 theta_serie_plot = None,
                 theta_serie_line_delimiter_plot = None,
                 theta_serie_plot_canvas = None
                 ):
        self.root = root
        
        self.data = data
        
        self.spinBoxWeek = spinBoxWeek
        self.displaying_week_was = displaying_week_was
        self.displaying_theta_was = displaying_theta_was
        
        self.time_serie_plot = time_serie_plot
        self.time_serie_plot_canvas = time_serie_plot_canvas
        
        self.samplingDropDownMenu = samplingDropDownMenu
        self.samplePeriod = samplePeriod
        self.windowSliderSelector = windowSliderSelector
        self.selectedWindow = selectedWindow
        self.red_lines_delimiters_plot = red_lines_delimiters_plot
        self.window_zoom_plot = window_zoom_plot
        
        self.spinBoxTheta = spinBoxTheta
        
        self.theta_serie_plot = theta_serie_plot
        self.theta_serie_plot_canvas = theta_serie_plot_canvas
        self.theta_serie_line_delimiter_plot = theta_serie_line_delimiter_plot
        
        
class GUI_Data():
    def __init__(self,
                 network_time_series=[],
                 theta_time_series=[],
                 Tsventana = 0,
                 n=0):
        self.all_series = network_time_series
        self.thetaParams = theta_time_series
        self.Tsventana = Tsventana
        self.n = n
    