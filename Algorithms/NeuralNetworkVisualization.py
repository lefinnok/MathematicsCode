# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 13:05:58 2021

@author: lefin
"""

#=====================GUI MODULE=======================#

import tkinter as tk
from tkinter import Tk, ttk, Button, LabelFrame, Frame, Toplevel, Canvas, Label, W, FLAT
import matplotlib.pyplot as plt

from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  
NavigationToolbar2Tk) 
import matplotlib
#matplotlib.use('TkAgg')

import numpy as np


from threading import Thread
from time import sleep
from NeuralNetwork import simple_network_model, activations

class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.title("Neural Network")
        self.state('zoomed')
        
        self.data = np.array(range(-100,100))
        self.batch = np.reshape(self.data,(200,1))    
        
        self.graph = Graph_Tl(self, self.data)
        
        self.canvas = ResizingCanvas(self,bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand = True)
        
        
        self.network = simple_network_model.network([1,8,8,1])
        self.network_update()
        
        self.pass_button = Button(self.canvas,text='pass',command = self.network_foward_pass,anchor=W)
        self.pass_button.configure(width = 10, activebackground = "#33B5E5",
                                                        relief = FLAT)
        self.pass_button.pack()
        
    def update_test(self):
        def counter():
            sleep(1)
            for n in np.arange(0,1.0,0.001)[::-1]:
                sleep(0.05)
                self.batch = activations.leakrelu(self.batch,leak=n)
                self.data = self.batch.reshape((200))
                self.graph.update_line(self.data)
            #print(self.graph.line)
        self.thread = Thread(target = counter)
        self.thread.start()
    
    def network_update(self):
        self.canvas.delete("all")
        self.canvas.clean_widgets()
        n_layer = len(self.network.layers)
        c_width,c_height = self.canvas.winfo_width(),self.canvas.winfo_height()
        
        l_width = c_width/n_layer #layer width is the canvas width over the number of layers
        node_r = min([min(c_height/(len(layer.nodes)+3),l_width)/2 for layer in self.network.layers])
        #print(c_width,c_height,l_width)
        for lidx,layer in enumerate(self.network.layers):
            layer_origin = lidx*l_width
            n_node = len(layer.nodes)
            layer_base = self.canvas.create_rectangle(layer_origin,0, layer_origin+l_width, c_height,fill='white')
            self.canvas.tag_lower(layer_base)
            node_space = c_height/(n_node+1)
            
            node_x = layer_origin+l_width/2
            
            self.canvas.create_text(node_x,5, text=f'layer {lidx}')
            
            for nidx,node in enumerate(layer.nodes,1): #set and draw nodes
                node_y = nidx*node_space
                self.canvas._create_circle(node_x,node_y,node_r, tags=('node'), fill='white')
                node.set_coordinates(node_x, node_y, node_r,self)
                if type(node.out_value) == int:
                    self.canvas.create_text(node_x,node_y,text = str(node.output_value), tags=('node_output'))
                
        
        for layer in self.network.layers: #draw connections and nodes
            for node in layer.nodes:
                for link in node.links:
                    self.canvas.create_line(node.x, node.y, node.links[link].targetNode.x, node.links[link].targetNode.y, tags=('link'))
        
        self.canvas.tag_raise('node','link')
        self.canvas.tag_raise('node_output','node')
                    
                
                
                
    def network_foward_pass(self):
        self.network.foward_pass()
    
    def start(self):
        self.network_foward_pass()
        self.mainloop()
    
        
class Graph_Tl(Toplevel):
    def __init__(self,master,init_plot):
        Toplevel.__init__(self,master) #create graph toplevel
        self.title('Output Graph') #set title
        self.attributes('-topmost', 'true') #toplevel always on top
        
        
        self.figure = Figure(figsize = (5,5), dpi = 100)
        self.subplot = self.figure.add_subplot(111)
        self.subplot.grid(True)
        self.line, = self.subplot.plot(init_plot)
        
        self.canvas = FigureCanvasTkAgg(self.figure,self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand = True)
        
        toolbar = NavigationToolbar2Tk(self.canvas,self) 
        toolbar.update() 

    def update_line(self, new_data):
        self.line.set_ydata(new_data)
        self.figure.canvas.draw_idle()
        
class ResizingCanvas(Canvas):
    def __init__(self,parent,**kwargs):
        Canvas.__init__(self,parent,**kwargs)
        self.parent = parent
        self.bind("<Configure>", self.on_resize)
        self.height = self.winfo_reqheight()
        self.width = self.winfo_reqwidth()
        self.widgets = []

    def on_resize(self,event):
        # determine the ratio of old width/height to new width/height
        wscale = float(event.width)/self.width
        hscale = float(event.height)/self.height
        self.width = event.width
        self.height = event.height
        # resize the canvas 
        self.config(width=self.width, height=self.height)
        # rescale all the objects tagged with the "all" tag
        self.scale("all",0,0,wscale,hscale)
        self.parent.network_update()

    def _create_circle(self, x, y, r, **kwargs):
        return self.create_oval(x-r, y-r, x+r, y+r, **kwargs)
    
    def create_button(self,x,y,**kwargs):
        button = Button(self, **kwargs)
        button.configure(height = 5,width = 10, activebackground = "#33B5E5")
        button.place(x=x,y=y)
        self.widget.append(button)
        button.pack()
    
    def clean_widgets(self):
        for n in range(self.widgets)-1:
            widget = self.widgets.pop(n)
            widget.destroy()
            
    
    
        

root = Root()

root.start()
