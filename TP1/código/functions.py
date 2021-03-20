import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
from scipy.fftpack import dct, idct
import itertools
import math
from sys import stdin,stdout

def encoder(image, q_factor=75, dsType='4:2:2', filt=False, BlockSize=8):
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    Y, Cb, Cr = RGB2YCbCr(R, G, B)

    Y, Cb_ds, Cr_ds = downsample(Y, Cb, Cr, dsType)

    Y_dct = dct_blocks(Y, BlockSize)
    Cb_dct = dct_blocks(Cb_ds, BlockSize)
    Cr_dct = dct_blocks(Cr_ds, BlockSize)
    
    Y_quant, Cb_quant, Cr_quant, q_cbcr, q_y = quantization(Y_dct, Cb_dct, Cr_dct, q_factor)
    
    Y_dpcm, Cb_dpcm, Cr_dpcm = dpcm(Y_quant, Cb_quant, Cr_quant)

    return Y_dpcm, Cb_dpcm, Cr_dpcm, q_cbcr, q_y

def decoder(Y_dpcm, Cb_dpcm, Cr_dpcm, q_cbcr, q_y, original_height, original_width, q_factor=75, dsType='4:2:2', filt=True, BlockSize=8):
    if(dsType == '4:2:2'):
        Cb_ds_height = original_height
        Cr_ds_height = original_height
        Cb_ds_width = int(original_width/2) + (original_width % 2)
        Cr_ds_width = int(original_width/2) + (original_width % 2)
    elif(dsType == '4:2:0'):
        Cb_ds_height = int(original_height/2) + (original_height % 2)
        Cr_ds_height = int(original_height/2) + (original_height % 2)
        Cb_ds_width = int(original_width/2) + (original_width % 2)
        Cr_ds_width = int(original_width/2) + (original_width % 2)
    elif(dsType == '4:4:4'):
        Cb_ds_height = original_height
        Cr_ds_height = original_height
        Cb_ds_width = original_width
        Cr_ds_width = original_width
    
    Y_quant, Cb_quant, Cr_quant = idpcm(Y_dpcm, Cb_dpcm, Cr_dpcm)
    
    Y_dct, Cb_dct, Cr_dct = iquantization(Y_quant, Cb_quant, Cr_quant, q_cbcr, q_y)

    Y = idct_blocks(Y_dct, original_height, original_width, BlockSize)
    Cb_ds = idct_blocks(Cb_dct, Cb_ds_height, Cb_ds_width, BlockSize)
    Cr_ds = idct_blocks(Cr_dct, Cr_ds_height, Cr_ds_width, BlockSize)

    Y, Cb, Cr = upsample(Y, Cb_ds, Cr_ds, dsType)

    R, G, B = YCbCr2RGB(Y, Cb, Cr)

    return R, G, B
    
### conversão de espaços de cor ###
def YCbCr2RGB(Y, Cb, Cr):
    height, width = Y.shape
    R = np.empty((height, width), dtype = np.uint8)
    G = np.empty((height, width), dtype = np.uint8)
    B = np.empty((height, width), dtype = np.uint8)

    R = Y + 1.402*(Cr - 128)
    G = Y - 0.344136*(Cb - 128) - 0.714136*(Cr - 128)
    B = Y + 1.771*(Cb - 128)

    R[R>255]=255
    R[R<0]=0

    G[G>255]=255
    G[G<0]=0

    B[B>255]=255
    B[B<0]=0

    R= np.round(R).astype(np.uint8)
    G= np.round(G).astype(np.uint8)
    B= np.round(B).astype(np.uint8)

    return R, G, B

def RGB2YCbCr(R, G, B):
    height, width = R.shape
    Y = np.empty((height, width), dtype = np.uint8)
    Cb = np.empty((height, width), dtype = np.uint8)
    Cr = np.empty((height, width), dtype = np.uint8)

    Y = 0.299*R + 0.587*G + 0.114*B
    Cb = (B - Y)/1.772 + 128
    Cr = (R - Y)/1.402 + 128

    return Y, Cb, Cr
### \conversão de espaços de cor ###
    
### downsampling ###
def downsample(C1, C2, C3, d):
    height, width = C1.shape
    if d == '4:2:0':
        C2 = C2[::2, ::2]
        C3 = C3[::2, ::2]
    elif d == '4:2:2':
        C2 = C2[:, ::2]
        C3 = C3[:, ::2]
    elif d == '4:4:4':
        pass

    return C1, C2, C3

def upsample(C1, C2, C3, d, filt=True):
	height, width = C1.shape
	if d == '4:2:0':
		C2_us = np.repeat(C2, 2, axis=1)
		C2_us = np.repeat(C2_us, 2, axis=0)
		C3_us = np.repeat(C3, 2, axis=1)
		C3_us = np.repeat(C3_us, 2, axis=0)
	elif d == '4:2:2':
		C2_us = np.repeat(C2, 2, axis=1)
		C3_us = np.repeat(C3, 2, axis=1)
	elif d == '4:4:4':
		pass

	return C1, C2_us[0:height, 0:width], C3_us[0:height, 0:width]
### \downsampling ###

### dct em blocos ###
def dct_2(canal):
    X_dct = dct(canal, norm='ortho').T
    dct_blocks = dct(X_dct,norm='ortho').T

    return dct_blocks

def idct_2(canal):
    X_idct = idct(canal, norm='ortho').T
    idct_blocks = idct(X_idct, norm='ortho').T
    return idct_blocks
	
def dct_blocks(canal, BlockSize=8):
    height, width = canal.shape
    column_padding_amount = (BlockSize - width % BlockSize) % BlockSize
    line_padding_amount = (BlockSize - height % BlockSize) % BlockSize

    column_padding = canal[:, width - 1].reshape(height, 1)    # extrair a última coluna
    column_padding = np.repeat(column_padding, column_padding_amount, axis=1)    # repetir
    canal = np.concatenate((canal, column_padding), axis=1)    # concatenar

    line_padding = canal[height - 1, :].reshape(1, width + column_padding_amount)    # extrair última linha
    line_padding = np.repeat(line_padding, line_padding_amount, axis=0)    # repetir
    canal = np.concatenate((canal, line_padding), axis=0)    # concatenar

    height, width = canal.shape
    n_blocks_height = int(height/BlockSize)
    n_blocks_width = int(width/BlockSize)

    for i in range(n_blocks_height):
        for j in range(n_blocks_width):
            canal[i*BlockSize:i*BlockSize + BlockSize, j*BlockSize:j*BlockSize + BlockSize] = \
                dct_2(canal[i*BlockSize:i*BlockSize + BlockSize, j*BlockSize:j*BlockSize + BlockSize])

    return canal

def idct_blocks(canal, original_height, original_width, BlockSize=8):
    height, width = canal.shape
    n_blocks_height = int(height/BlockSize)
    n_blocks_width = int(width/BlockSize)

    for i in range(n_blocks_height):
        for j in range(n_blocks_width):
            canal[i*BlockSize:i*BlockSize + BlockSize, j*BlockSize:j*BlockSize + BlockSize] = \
                idct_2(canal[i*BlockSize:i*BlockSize + BlockSize, j*BlockSize:j*BlockSize + BlockSize])

    canal = canal[0:original_height, 0:original_width]

    return canal
### \dct em blocos ###

### quantização ###
def quantization(Y_dct, Cb_dct, Cr_dct,fact_q):    
    q_cbcr = np.array([17,18,24,47,99,99,99,99,\
                18,21,26,66,99,99,99,99,    \
                24,26,56,99,99,99,99,99,    \
                47,66,99,99,99,99,99,99,    \
                99,99,99,99,99,99,99,99,    \
                99,99,99,99,99,99,99,99,    \
                99,99,99,99,99,99,99,99,    \
                99,99,99,99,99,99,99,99]).reshape(8,8)
    
    q_y = np.array([16,11,10,16,24,40,51,61,\
                12,12,14,19,26,58,60,55,	\
                14,13,16,24,40,57,69,56,	\
                14,17,22,29,51,87,80,62,	\
                18,22,37,56,68,109,103,77,	\
                24,35,55,64,81,104,113,92,	\
                49,64,78,87,103,121,120,101,\
                72,92,95,98,112,100,103,99]).reshape(8,8)
        
    if (fact_q >= 50):
        sf = (100 - fact_q) / 50
    else:
        sf = 50 / fact_q

    if (sf  == 0 ):
        q_cbcr= np.round(q_cbcr/q_cbcr) #.astype(np.uint8)
        q_y= np.round(q_y/q_y) #.astype(np.uint8)
    else:
        q_cbcr= np.round(q_cbcr*sf) #.astype(np.uint8)
        q_y = np.round(q_y*sf) #.astype(np.uint8)
        
    q_cbcr[q_cbcr>255]=255
    q_y[q_y>255]=255
    
    height_Y, width_Y = Y_dct.shape
    height_C, width_C = Cb_dct.shape
    Y_quant = np.empty((height_Y,width_Y), dtype='int16')
    Cb_quant = np.empty((height_C,width_C), dtype='int16')
    Cr_quant = np.empty((height_C,width_C), dtype='int16')

    for i in range(0,height_Y,8):
       for j in range(0,width_Y,8):
          Y_quant[i:i+8, j:j+8] = np.round(Y_dct[i:i+8, j:j+8] / q_y)
          if(i < height_C and j < width_C):
               Cb_quant[i:i+8, j:j+8] = np.round(Cb_dct[i:i+8,j:j+8] / q_cbcr)
               Cr_quant[i:i+8, j:j+8] = np.round(Cr_dct[i:i+8,j:j+8] / q_cbcr)
    
    return Y_quant, Cb_quant, Cr_quant, q_cbcr, q_y

def outln(n):
	stdout.write(str(n))
	stdout.write("\n")

def iquantization(Y_quant, Cb_quant, Cr_quant, q_cbcr, q_y):
    height_Y, width_Y = Y_quant.shape
    height_C, width_C = Cb_quant.shape
    Y_iq = np.empty((height_Y,width_Y))
    print(Y_iq.dtype)
    Cb_iq = np.empty((height_C,width_C))
    Cr_iq = np.empty((height_C,width_C))
    
    for i in range(0,height_Y,8):
      for j in range(0,width_Y,8):
         Y_iq[i:i+8, j:j+8] = Y_quant[i:i+8, j:j+8] * q_y
         if(i < height_C and j < width_C):
              Cb_iq[i:i+8, j:j+8] = Cb_quant[i:i+8,j:j+8] * q_cbcr
              Cr_iq[i:i+8, j:j+8] = Cr_quant[i:i+8,j:j+8] * q_cbcr
   
    return Y_iq, Cb_iq, Cr_iq
### \quantização ###

### dpcm ###
def dpcm(Y_quant, Cb_quant, Cr_quant):
	height_y, width_y = Y_quant.shape
	height_c, width_c = Cb_quant.shape

	Y_dpcm = np.copy(Y_quant)
	Cb_dpcm = np.copy(Cb_quant)
	Cr_dpcm = np.copy(Cr_quant)

	for i in range(8, width_y, 8):
		Y_dpcm[0, i] = Y_quant[0, i] - Y_quant[0, i - 8]
		if(i < width_c):
			Cb_dpcm[0, i] = Cb_quant[0, i] - Cb_quant[0, i - 8]
			Cr_dpcm[0, i] = Cr_quant[0, i] - Cr_quant[0, i - 8]
		
	for i in range(8, height_y, 8):
		Y_dpcm[i, 0] = Y_quant[i, 0] - Y_quant[i - 8, width_y - 8]
		if(i < height_c):
			Cb_dpcm[i, 0] = Cb_quant[i, 0] - Cb_quant[i - 8, width_c - 8]
			Cr_dpcm[i, 0] = Cr_quant[i, 0] - Cr_quant[i - 8, width_c - 8]
		for j in range(8, width_y, 8):
		    Y_dpcm[i, j] = Y_quant[i, j] - Y_quant[i, j - 8]
		    if(i < height_c and j < width_c):
		        Cb_dpcm[i, j] = Cb_quant[i, j] - Cb_quant[i, j - 8]
		        Cr_dpcm[i, j] = Cr_quant[i, j] - Cr_quant[i, j - 8]

	return Y_dpcm, Cb_dpcm, Cr_dpcm

def idpcm(Y_dpcm, Cb_dpcm, Cr_dpcm):
    height_y, width_y = Y_dpcm.shape
    height_c, width_c = Cb_dpcm.shape

    Y_quant = np.copy(Y_dpcm)
    Cb_quant = np.copy(Cb_dpcm)
    Cr_quant = np.copy(Cr_dpcm)
    
    for i in range(8, width_y, 8):
    	Y_quant[0, i] = Y_dpcm[0, i] + Y_quant[0, i - 8]
    	if(i < width_c):
    		Cb_quant[0, i] = Cb_dpcm[0, i] + Cb_quant[0, i - 8]
    		Cr_quant[0, i] = Cr_dpcm[0, i] + Cr_quant[0, i - 8]
                
    for i in range(8, height_y, 8):
    	Y_quant[i, 0] = Y_dpcm[i, 0] + Y_quant[i - 8, width_y - 8]
    	if(i < height_c):
    		Cb_quant[i, 0] = Cb_dpcm[i, 0] + Cb_quant[i - 8, width_c - 8]
    		Cr_quant[i, 0] = Cr_dpcm[i, 0] + Cr_quant[i - 8, width_c - 8]
    	for j in range(8, width_y, 8):
            Y_quant[i, j] = Y_dpcm[i, j] + Y_quant[i - 8, (j - 8)]
            if(i < height_c and j < width_c):
                Cb_quant[i, j] = Cb_dpcm[i, j] + Cb_quant[i - 8, (j - 8)]
                Cr_quant[i, j] = Cr_dpcm[i, j] + Cr_quant[i - 8, (j - 8)]
                
    return Y_quant, Cb_quant, Cr_quant
### \dpcm ###
