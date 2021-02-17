#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 10:12:32 2021

@author: etiandro
"""


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as clr

img = mpimg.imread('peppers.bmp')

def ex2(img,colormap=False): 
    #Funcao geral para plot imagens
    plt.figure()
    plt.axis('off')
    plt.imshow(img, cmap=colormap)
    plt.show()

#1.1
def ex2_1(colors, name, image): 
    #Funcao para criar um colormap, basta usar como argumento a matriz das cores, o nome do colormap e a imagem em questao
    colormap = clr.LinearSegmentedColormap.from_list(name, colors, N=256)
    ex2(image,colormap)

def aux():
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    canais = [R,G,B]
    
    return canais

def ex2_2(img):
    #Visualizacao da imagem nos 3 canais RGB individuais
    canal = aux()
   
    colors_red = [(0,0,0), (1,0,0)]
    colors_green = [(0,0,0), (0,1,0)]
    colors_blue = [(0,0,0), (0,0,1)]
   
    
    ex2(img,None)
    ex2_1(colors_red,"Red", canal[0])
    ex2_1(colors_green,"Green", canal[1])
    ex2_1(colors_blue, "Blue", canal[2])
    
    

def ex3(img):
    #Conversao de imagem do modelo de cor RGB pafa o modelo YCbCr
    canal = aux()
    colors_gray =[(0,0,0),(0.5,0.5,0.5)]
    
    Y = 0.299*canal[0] + 0.587*canal[1] + 0.114*canal[2]
    Cb = ((canal[2] - Y)/1.772)+128
    Cr = ((canal[0] - Y)/1.402)+128
    
    #print das imagens com o modelo de cor YCbCr
    ex2_1(colors_gray,"Y Gray", Y)
    ex2_1(colors_gray,"Cb Gray", Cb)
    ex2_1(colors_gray, "Cr Gray", Cr)
    
if __name__ == "__main__":
    ex2_2(img)
    ex3(img)
    
    
    
    