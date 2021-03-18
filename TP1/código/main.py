from functions import *

def open_image(path):
    img = plt.imread(path)

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    return img, R, G, B
    
def view(colors, name, image):
    colormap = clr.LinearSegmentedColormap.from_list(name, colors, N=256)
    plt.figure()
    plt.axis('off')
    plt.imshow(image, cmap = colormap)


def view_dct(image):
    colors= [(0,0,0), (0.5,0.5,0.5)]
    colormap = clr.LinearSegmentedColormap.from_list("dct YCBCR", colors, N=256)
    plt.figure()
    plt.imshow(np.log(abs(image) + 0.0001), cmap = colormap)

def ex2():
    img, R, G, B = open_image('../imagens/peppers.bmp')
    plt.figure()
    plt.axis('off')
    plt.imshow(img)

    colors_red = [(0,0,0), (1,0,0)]
    colors_green = [(0,0,0), (0,1,0)]
    colors_blue = [(0,0,0), (0,0,1)]

    view(colors_red, 'red', R)
    view(colors_green, 'green', G)
    view(colors_blue, 'blue', B)

    plt.show()

def ex3():
    img, R, G, B = open_image('../imagens/peppers.bmp')

    colors_grayscale = [(0,0,0), (0.5,0.5,0.5)]
    colors_red = [(0,0,0), (1,0,0)]
    colors_green = [(0,0,0), (0,1,0)]
    colors_blue = [(0,0,0), (0,0,1)]

    Y, Cb, Cr = RGB2YCbCr(R, G, B)

    view(colors_grayscale, 'grayscale', Y)
    view(colors_grayscale, 'blue', Cb)
    view(colors_grayscale, 'red', Cr)

    R2, G2, B2 = YCbCr2RGB(Y, Cb, Cr)
    #testar a conversao se deu os valores originais
    #print(R2 == R)
    view(colors_red, 'red', R2)
    view(colors_green, 'green', G2)
    view(colors_blue, 'blue', B2)

    plt.show()

def ex4():
    img, R, G, B = open_image('../imagens/peppers.bmp')
    print("Dimensões originais:", img.shape)

    plt.figure()
    plt.imshow(img)

    ds = '4:2:0'

    Y, Cb, Cr = RGB2YCbCr(R, G, B)

    Y, Cb_d, Cr_d = downsample(Y, Cb, Cr, ds)

    print("Dimensões Y:", Y.shape)
    print("Dimensões Cb_d:", Cb_d.shape)
    print("Dimensões Cr_d:", Cr_d.shape)

    #colors_grayscale = [(0,0,0), (0.5,0.5,0.5)]
    #view(colors_grayscale, 'grayscale', Y)
    #view(colors_grayscale, 'grayscale', Cb_d)
    #view(colors_grayscale, 'grayscale', Cr_d)

    Y_r, Cb_r, Cd_r = upsample(Y, Cb_d, Cr_d, ds)
    R_r, G_r, B_r = YCbCr2RGB(Y_r, Cb_r, Cd_r)

    img_r = img.copy()
    img_r[:,:,0] = R_r
    img_r[:,:,1] = G_r
    img_r[:,:,2] = B_r

    plt.figure()
    plt.imshow(img_r, None)

    plt.show()

    return Y, Cb_d, Cr_d

def ex5_1():
    Y, Cb_d, Cr_d = ex4()

    Y_dct = dct_2(Y)
    Cb_dct = dct_2(Cb_d)
    Cr_dct = dct_2(Cr_d)

    view_dct(Y_dct)
    view_dct(Cb_dct)
    view_dct(Cr_dct)

    Y_idct = idct_2(Y_dct)
    Cb_idct = idct_2(Cb_dct)
    Cr_idct = idct_2(Cr_dct)

    colors_grayscale = [(0,0,0), (0.5,0.5,0.5)]
    view(colors_grayscale, 'grayscale', Y_idct)
    view(colors_grayscale, 'grayscale', Cb_idct)
    view(colors_grayscale, 'grayscale', Cr_idct)

    plt.show()

def ex5_23(bs=8):
    img, R, G, B = open_image('../imagens/peppers.bmp')
    colors_grayscale = [(0,0,0), (0.5,0.5,0.5)]

    ds = '4:2:0'
    filt = False

    Y, Cb, Cr = RGB2YCbCr(R, G, B)

    Y, Cb_ds, Cr_ds = downsample(Y, Cb, Cr, ds)

    Y_dct = dct_blocks(Y, bs)
    Cb_dct = dct_blocks(Cb_ds, bs)
    Cr_dct = dct_blocks(Cr_ds, bs)
    
   
    view_dct(Y_dct)
    view_dct(Cb_dct)
    view_dct(Cr_dct)

    height, width = R.shape

    Y_r = idct_blocks(Y_dct, height, width, bs)
    Cb_ds_r = idct_blocks(Cb_dct, int(height/2), int(width/2), bs)
    Cr_ds_r = idct_blocks(Cr_dct, int(height/2), int(width/2), bs)

    #Quantizacao 

    [Y_quant, Cb_quant, Cr_quant,q_cbcr,q_y]= quantization(Y_dct, Cb_dct, Cr_dct,100)
    view_dct(Y_quant)
    view_dct(Cb_quant)
    view_dct(Cr_quant)
    
    #Quantizacao inversa 
    [Y_iquant,Cb_iquant,Cr_iquant]= iquantization(Y_quant, Cb_quant, Cr_quant, q_cbcr, q_y)
    view_dct(Y_iquant)
    view_dct(Cb_iquant)
    view_dct(Cr_iquant)

    print("Y:", Y)
    print("Y_r", Y_r)

    plt.show()
    
def ex6():
	img, R, G, B = open_image('../imagens/peppers.bmp')
	colors_grayscale = [(0,0,0), (0.5,0.5,0.5)]

	ds = '4:2:0'
	filt = False
	bs = 8

	Y, Cb, Cr = RGB2YCbCr(R, G, B)
	Y, Cb_ds, Cr_ds = downsample(Y, Cb, Cr, ds)

	Y_dct = dct_blocks(Y, bs)
	Cb_dct = dct_blocks(Cb_ds, bs)
	Cr_dct = dct_blocks(Cr_ds, bs)

	view_dct(Y_dct)
	#view_dct(Cb_dct)
	#view_dct(Cr_dct)

	#Quantizacao 
	Y_quant, Cb_quant, Cr_quant, q_cbcr, q_y = quantization(Y_dct, Cb_dct, Cr_dct, 50)
	view_dct(Y_quant)
	#view_dct(Cb_quant)
	#view_dct(Cr_quant)

	#Quantizacao inversa 
	Y_iquant, Cb_iquant, Cr_iquant = iquantization(Y_quant, Cb_quant, Cr_quant, q_cbcr, q_y)
	view_dct(Y_iquant)
	#view_dct(Cb_iquant)
	#view_dct(Cr_iquant)
	
	height, width = R.shape

	Y_r = idct_blocks(Y_iquant, height, width, bs)
	Cb_ds_r = idct_blocks(Cb_iquant, int(height/2), int(width/2), bs)
	Cr_ds_r = idct_blocks(Cr_iquant, int(height/2), int(width/2), bs)
	
	view(colors_grayscale, 'grayscale', Y_r)
	
	Y_r, Cb_r, Cr_r = upsample(Y_r, Cb_ds_r, Cr_ds_r, ds)
	
	R_r, G_r, B_r = YCbCr2RGB(Y_r, Cb_r, Cr_r)
	
	img_r = img.copy();
	img_r[:,:,0] = R_r
	img_r[:,:,1] = G_r
	img_r[:,:,2] = B_r
	
	#plt.imshow(img_r)

	print("Y:", Y_dct[200:208, 200:208])
	print("Y_r", Y_iquant[200:208, 200:208])

	plt.show()
	
def ex7():
	img, R, G, B = open_image('../imagens/peppers.bmp')
	colors_grayscale = [(0,0,0), (0.5,0.5,0.5)]
	
	height, width = R.shape

	ds = '4:2:0'
	filt = False
	bs = 8

	Y, Cb, Cr = RGB2YCbCr(R, G, B)
	Y, Cb_ds, Cr_ds = downsample(Y, Cb, Cr, ds)

	Y_dct = dct_blocks(Y, bs)
	Cb_dct = dct_blocks(Cb_ds, bs)
	Cr_dct = dct_blocks(Cr_ds, bs)
	
	Y_quant, Cb_quant, Cr_quant, q_cbcr, q_y = quantization(Y_dct, Cb_dct, Cr_dct, 100)
	
	Y_dpcm, Cb_dpcm, Cr_dpcm = dpcm(Y_quant, Cb_quant, Cr_quant)
	
	Y_quant_r, Cb_quant_r, Cr_quant_r = idpcm(Y_dpcm, Cb_dpcm, Cr_dpcm)
	
	Y_dct_r, Cb_dct_r, Cr_dct_r = iquantization(Y_quant_r, Cb_quant_r, Cr_quant_r, q_cbcr, q_y)

	view_dct(Y_quant[8:24, 8:24])
	view_dct(Y_dpcm[8:24, 8:24])
	print("Y_dct", Y_quant[8:16, 8:16])
	print("Y_dpcm", Y_dpcm[8:16, 8:16])
	
	Y_r = idct_blocks(Y_dct_r, height, width)
	Cb_ds_r = idct_blocks(Cb_dct_r, height, width)
	Cr_ds_r = idct_blocks(Cr_dct_r, height, width)
	
	Y_r, Cb_r, Cr_r = upsample(Y_r, Cb_ds_r, Cr_ds_r, ds)
	
	R_r, G_r, B_r = YCbCr2RGB(Y_r, Cb_r, Cr_r)
	
	img_r = img.copy()
	img_r[:,:,0] = R_r
	img_r[:,:,1] = G_r
	img_r[:,:,2] = B_r
	
	plt.figure()
	plt.imshow(img_r)
	
	plt.show()

if __name__ == "__main__":
	#ex2()
	#ex3()
	#ex4()
	#ex5_1()
	#ex5_23(8)
	#ex5_23(64)
	#ex6()
	#ex7()

	ds = '4:2:0'
	q_factor = 100
	f = False
	bs = 8

	img = plt.imread('../imagens/logo.bmp')

	height,width = img[:,:,0].shape
	# encoder(image, q_factor=75, dsType='4:2:2', filt=False, BlockSize=8)        
	Y_dpcm, Cb_dpcm, Cr_dpcm, q_cbcr, q_y = encoder(img, q_factor, ds, f, bs)

	# decoder(Y_dpcm, Cb_dpcm, Cr_dpcm, q_cbcr, q_y, original_height, original_width, q_factor=75, dsType='4:2:2', filt=True, BlockSize=8)
	R_r, G_r, B_r = decoder(Y_dpcm, Cb_dpcm, Cr_dpcm, q_cbcr, q_y, height, width, q_factor, ds, f, bs)

	img_r = img.copy()
	img_r[:,:,0] = R_r
	img_r[:,:,1] = G_r
	img_r[:,:,2] = B_r

	plt.figure()
	plt.imshow(img)
	plt.figure()
	plt.imshow(img_r)

	plt.show()

