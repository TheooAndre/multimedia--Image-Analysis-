import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np

def open_image(path):
    img = plt.imread(path)

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    return img, R, G, B

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

def view(colors, name, image):
    colormap = clr.LinearSegmentedColormap.from_list(name, colors, N=256)
    plt.figure()
    plt.imshow(image, cmap = colormap)

def ex3():
    img, R, G, B = open_image('../imagens/peppers.bmp')

    colors_grayscale = [(0,0,0), (0.5,0.5,0.5)]
    colors_red = [(0,0,0), (1,0,0)]
    colors_green = [(0,0,0), (0,1,0)]
    colors_blue = [(0,0,0), (0,0,1)]

    plt.figure()
    plt.imshow(img)
    view(colors_red, 'red', R)
    view(colors_green, 'green', G)
    view(colors_blue, 'blue', B)

    Y, Cb, Cr = RGB2YCbCr(R, G, B)

    view(colors_grayscale, 'grayscale', Y)
    view(colors_grayscale, 'blue', Cb)
    view(colors_grayscale, 'red', Cr)

    R2, G2, B2 = YCbCr2RGB(Y, Cb, Cr);
   #testar a conversao se deu os valores originais
   # print(R2 == R)
    view(colors_red, 'red', R2)
    view(colors_green, 'green', G2)
    view(colors_blue, 'blue', B2)

    plt.show()

def YCbCr2RGB(Y, Cb, Cr):
    #R = Y + 1.402*(Cr - 128)
    #G = Y - 0.344136*(Cb - 128) - 0.714136*(Cr - 128)
    #B = Y + 1.772*(Cb - 128)
    height, width = Y.shape
    R = np.empty((height, width), dtype = np.int8)
    G = np.empty((height, width), dtype = np.int8)
    B = np.empty((height, width), dtype = np.int8)

    R = Y + 1.402*(Cr - 128)
    G = Y - 0.344136*(Cb - 128) - 0.714136*(Cr - 128)
    B = Y + 1.771*(Cb - 128)
    
    R= np.round(R).astype(np.uint8)
    G= np.round(G).astype(np.uint8)
    B= np.round(B).astype(np.uint8)
    
    R[R>255]=255
    R[R<0]=0
    
    G[G>255]=255
    G[G<0]=0
    
    B[B>255]=255
    B[B<0]=0

    return R, G, B

def RGB2YCbCr(R, G, B):
    #Y = 0.299*R + 0.587*G + 0.114*B
    #Cb = (B-Y)/1.772 + 128
    #Cr = (R-Y)/1.402 + 128
    height, width = R.shape
    Y = np.empty((height, width), dtype = np.int8)
    Cb = np.empty((height, width), dtype = np.int8)
    Cr = np.empty((height, width), dtype = np.int8)
    '''
    Y = 0.299*R + 0.587*G + 0.114*B
    Cb = -0.168736*R - 0.331264*G + 0.5*B + 256
    Cr = 0.5*R - 0.418688*G - 0.081312*B + 256
    '''
    Y = 0.299*R + 0.587*G + 0.114*B
    Cb = (B - Y)/1.772 + 128
    Cr = (R - Y)/1.402 + 128
    
    return Y, Cb, Cr

def ex4():
	img, R, G, B = open_image('../imagens/peppers.bmp')
	
	ds = '4:2:2'
	f = false
	bs = 8
	
	Y, Cb_ds, Cr_ds = encoder(img, ds, f, bs)
	
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
    
def encoder(image, dsType='4:2:2', filt=false, BlockSize=8):
	R = image[:, :, 0]
	G = image[:, :, 1]
	B = image[:, :, 2]
	
	Y, Cb, Cr = RGB2YCbCr(R, G, B)
	
	Y, Cb_ds, Cr_ds = downsample(Y, Cb, Cr, dsType)
	
	return Y, Cb_ds, Cr_ds
	
if __name__ == "__main__":
	#ex2()
	#ex3()
	ex4()

