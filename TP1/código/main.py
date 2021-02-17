import matplotlib.pyplot as plt
import matplotlib.colors as clr

def ex2():
	img = plt.imread('../imagens/peppers.bmp')
	plt.figure()
	plt.axis('off')
	plt.imshow(img)
	

	R = img[:, :, 0]
	G = img[:, :, 1]
	B = img[:, :, 2]

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

if __name__ == "__main__":
	ex2()
