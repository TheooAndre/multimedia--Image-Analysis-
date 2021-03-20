from functions import *

images = ['peppers.bmp', 'barn_mountains.bmp', 'logo.bmp']

colors_red = [(0, 0, 0), (1, 0, 0)]
colors_green = [(0, 0, 0), (0, 1, 0)]
colors_blue = [(0, 0, 0), (0, 0, 1)]
colors_grayscale = [(0,0,0), (1, 1, 1)]

### ex2 ###
def ex2():
	img = plt.imread('../imagens/peppers.bmp')
	
	R = img[:,:,0]
	G = img[:,:,1]
	B = img[:,:,2]
	
	view2(colors_red, 'red', R, 'Red channel')
	view2(colors_green, 'green', G, 'Green channel')
	view2(colors_blue, 'blue', B, 'Blue channel')
	
	plt.show()
### \ex2 ###

### ex3 ###
def ex3():	
	for i in images:
		img = plt.imread('../imagens/' + i)
		R = img[:,:,0]
		G = img[:,:,1]
		B = img[:,:,2]
		Y, Cb, Cr = RGB2YCbCr(R, G, B)
		view2(colors_grayscale, 'gray', Y, 'Y channel ' + i)
		view2(colors_red, 'red', R, 'Red channel ' + i)
		view2(colors_green, 'green', G, 'Green channel ' + i)
		view2(colors_blue, 'blue', B, 'Blue channel ' + i)
		view2(colors_grayscale, 'gray', Cb, 'Cb channel ' + i)
		view2(colors_grayscale, 'gray', Cr, 'Cr channel ' + i)
		plt.show()
### \ex3 ###

### ex4 ###
def ex4():
	for i in images:
		img = plt.imread('../imagens/' + i)
		R = img[:,:,0]
		G = img[:,:,1]
		B = img[:,:,2]
		Y, Cb, Cr = RGB2YCbCr(R, G, B)
		Y, Cb_ds, Cr_ds = downsample(Y, Cb, Cr, '4:2:0')
		view2(colors_grayscale, 'gray', Y, 'Y_ds ' + i)
		view2(colors_grayscale, 'gray', Cb_ds, 'Cb_ds ' + i)
		view2(colors_grayscale, 'gray', Cr_ds, 'Cr_ds ' + i)
		print('Dimensões Y_ds ' + i +':', Y.shape)
		print('Dimensões Cb_ds ' + i + ':', Cb_ds.shape)
		print('Dimensões Cr_ds ' + i + ':', Cr_ds.shape)
		plt.show()
### \ex4 ###

### ex5 ###
def ex5():
	for i in images:
		img = plt.imread('../imagens/' + i)
		R = img[:,:,0]
		G = img[:,:,1]
		B = img[:,:,2]
		Y, Cb, Cr = RGB2YCbCr(R, G, B)
		Y, Cb_ds, Cr_ds = downsample(Y, Cb, Cr, '4:2:0')
		Y_dct = dct_2(Y)
		Cb_dct = dct_2(Cb_ds)
		Cr_dct = dct_2(Cr_ds)
		view_dct2(Y_dct, 'Y_dct ' + i)
		view_dct2(Cb_dct, 'Cb_dct ' + i)
		view_dct2(Cr_dct, 'Cr_dct ' + i)
		plt.show()

def ex5_blocks(bs):
	for i in images:
		img = plt.imread('../imagens/' + i)
		R = img[:,:,0]
		G = img[:,:,1]
		B = img[:,:,2]
		Y, Cb, Cr = RGB2YCbCr(R, G, B)
		Y, Cb_ds, Cr_ds = downsample(Y, Cb, Cr, '4:2:0')
		Y_dct = dct_blocks(Y, bs)
		Cb_dct = dct_blocks(Cb_ds, bs)
		Cr_dct = dct_blocks(Cr_ds, bs)
		view_dct2(Y_dct, 'Y_dct ' + str(bs) + ' '+ i)
		view_dct2(Cb_dct, 'Cb_dct ' + str(bs) + ' '+ i)
		view_dct2(Cr_dct, 'Cr_dct ' + str(bs) + ' '+ i)
		plt.show()
### \ex5 ###

### ex6 ###
def ex6():
	for i in images:
		img = plt.imread('../imagens/' + i)
		R = img[:,:,0]
		G = img[:,:,1]
		B = img[:,:,2]
		Y, Cb, Cr = RGB2YCbCr(R, G, B)
		Y, Cb_ds, Cr_ds = downsample(Y, Cb, Cr, '4:2:0')
		Y_dct = dct_blocks(Y, 8)
		Cb_dct = dct_blocks(Cb_ds, 8)
		Cr_dct = dct_blocks(Cr_ds, 8)
		
		Y_quant10, _, _, _, _ = quantization(Y_dct, Cb_dct, Cr_dct, 10)
		Y_quant25, _, _, _, _ = quantization(Y_dct, Cb_dct, Cr_dct, 25)
		Y_quant50, _, _, _, _ = quantization(Y_dct, Cb_dct, Cr_dct, 50)
		Y_quant75, _, _, _, _ = quantization(Y_dct, Cb_dct, Cr_dct, 75)
		Y_quant100, _, _, _, _ = quantization(Y_dct, Cb_dct, Cr_dct, 100)
		
		view_dct2(Y_dct[0:16,0:16], 'Y_dct ' + i)
		view_dct2(Y_quant10[0:16,0:16], 'Y_quant 10 ' + i)
		view_dct2(Y_quant25[0:16,0:16], 'Y_quant 25 ' + i)
		view_dct2(Y_quant50[0:16,0:16], 'Y_quant 50 ' + i)
		view_dct2(Y_quant75[0:16,0:16], 'Y_quant 75 ' + i)
		view_dct2(Y_quant100[0:16,0:16], 'Y_quant 100 ' + i)
		plt.show()
### \ex6 ###

### ex7 ###
def ex7():
	for i in images:
		img = plt.imread('../imagens/' + i)
		R = img[:,:,0]
		G = img[:,:,1]
		B = img[:,:,2]
		Y, Cb, Cr = RGB2YCbCr(R, G, B)
		Y, Cb_ds, Cr_ds = downsample(Y, Cb, Cr, '4:2:0')
		Y_dct = dct_blocks(Y, 8)
		Cb_dct = dct_blocks(Cb_ds, 8)
		Cr_dct = dct_blocks(Cr_ds, 8)
		Y_quant100, Cb_quant100, Cr_quant100, _, _ = quantization(Y_dct, Cb_dct, Cr_dct, 100)
		
		Y_dpcm, Cb_dpcm, Cr_dpcm = dpcm(Y_quant100, Cb_quant100, Cr_quant100)
		view_dct2(Y_quant100[0:16,0:16], 'Y_quant 100 ' + i)
		view_dct2(Y_dpcm[0:16,0:16], 'Y_dpcm ' + i)
		plt.show()
### \ex7 ###

### ex8 ###
def ex8():
	for i in images:
		img = plt.imread('../imagens/' + i)
		height, width = img[:,:,0].shape
		
		Y_dpcm10, Cb_dpcm10, Cr_dpcm10, q_cbcr10, q_y10 = encoder(img, 10, '4:2:0')
		R10, G10, B10 = decoder(Y_dpcm10, Cb_dpcm10, Cr_dpcm10, q_cbcr10, q_y10, height, width, 75, '4:2:0')
		Y_dpcm25, Cb_dpcm25, Cr_dpcm25, q_cbcr25, q_y25 = encoder(img, 25, '4:2:0')
		R25, G25, B25 = decoder(Y_dpcm25, Cb_dpcm25, Cr_dpcm25, q_cbcr25, q_y25, height, width, 75, '4:2:0')
		Y_dpcm50, Cb_dpcm50, Cr_dpcm50, q_cbcr50, q_y50 = encoder(img, 50, '4:2:0')
		R50, G50, B50 = decoder(Y_dpcm50, Cb_dpcm50, Cr_dpcm50, q_cbcr50, q_y50, height, width, 75, '4:2:0')
		Y_dpcm75, Cb_dpcm75, Cr_dpcm75, q_cbcr75, q_y75 = encoder(img, 75, '4:2:0')
		R75, G75, B75 = decoder(Y_dpcm75, Cb_dpcm75, Cr_dpcm75, q_cbcr75, q_y75, height, width, 75, '4:2:0')
		Y_dpcm100, Cb_dpcm100, Cr_dpcm100, q_cbcr100, q_y100 = encoder(img, 100, '4:2:0')
		R100, G100, B100 = decoder(Y_dpcm100, Cb_dpcm100, Cr_dpcm100, q_cbcr100, q_y100, height, width, 75, '4:2:0')
		
		img_r10 = np.empty(img.shape, dtype='uint8')
		img_r10[:,:,0] = R10;
		img_r10[:,:,1] = G10;
		img_r10[:,:,2] = B10;
		img_r25 = np.empty(img.shape, dtype='uint8')
		img_r25[:,:,0] = R25;
		img_r25[:,:,1] = G25;
		img_r25[:,:,2] = B25;
		img_r50 = np.empty(img.shape, dtype='uint8')
		img_r50[:,:,0] = R50;
		img_r50[:,:,1] = G50;
		img_r50[:,:,2] = B50;
		img_r75 = np.empty(img.shape, dtype='uint8')
		img_r75[:,:,0] = R75;
		img_r75[:,:,1] = G75;
		img_r75[:,:,2] = B75;
		img_r100 = np.empty(img.shape, dtype='uint8')
		img_r100[:,:,0] = R100;
		img_r100[:,:,1] = G100;
		img_r100[:,:,2] = B100;
		
		plt.figure()
		plt.title('Imagem original')
		plt.imshow(img)
		plt.figure()
		plt.title('Imagem reconstruida 10')
		plt.imshow(img_r10)
		plt.figure()
		plt.title('Imagem reconstruida 25')
		plt.imshow(img_r25)
		plt.figure()
		plt.title('Imagem reconstruida 50')
		plt.imshow(img_r50)
		plt.figure()
		plt.title('Imagem reconstruida 75')
		plt.imshow(img_r75)
		plt.figure()
		plt.title('Imagem reconstruida 100')
		plt.imshow(img_r100)
		plt.show()
### \ex8 ###
	
if __name__ == "__main__":
	ex2()
	print("Press any key to continue")
	input()
	ex3()
	print("Press any key to continue")
	input()
	ex4()
	print("Press any key to continue")
	input()
	ex5()
	print("Press any key to continue")
	input()
	ex5_blocks(8)
	print("Press any key to continue")
	input()
	ex5_blocks(64)
	print("Press any key to continue")
	input()
	ex6()
	print("Press any key to continue")
	input()
	ex7()
	print("Press any key to continue")
	input()
	ex8()
	print("Press any key to exit")
	input()
