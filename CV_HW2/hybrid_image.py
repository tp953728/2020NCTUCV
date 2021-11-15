import cv2
import numpy as np
import math
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt


img_list = [('0_Afghan_girl_before.jpg','0_Afghan_girl_after.jpg'),
			('1_bicycle.bmp','1_motorcycle.bmp'),
			('2_bird.bmp','2_plane.bmp'),
			('3_cat.bmp','3_dog.bmp'),
			('4_einstein.bmp','4_marilyn.bmp'),
			('5_fish.bmp','5_submarine.bmp'),
			('6_makeup_before.jpg','6_makeup_after.jpg'),
			('7_girl.jpg','7_man.jpg'),
			('8_pooh.jpg','8_xi.jpg'),
			('9_bulb.jpg','9_ice_cream.jpg'),
			('11_lion.jpg','11_cat.jpg')]

def GaussianFilter(x, y, sigma, lowpass):
	kernel = np.zeros((x,y))
	center_x = int(x/2) if x%2 == 0 else int(x/2) + 1
	center_y = int(y/2) if y%2 == 0 else int(y/2) + 1

	for i in range(x):
		for j in range(y):
			kernel[i][j] = math.exp(-1.0 * ((i - center_x)**2 + (j - center_y)**2) / (2 * sigma ** 2))
	return kernel if lowpass else 1-kernel 

def IdealFilter(x, y, cutoff, lowpass):
	kernel = np.zeros((x,y))
	center_x = int(x/2) if x%2 == 0 else int(x/2) + 1
	center_y = int(y/2) if y%2 == 0 else int(y/2) + 1

	for i in range(x):
		for j in range(y):
			D = math.sqrt((i-center_x)**2 + (j-center_y)**2)
			kernel[i][j] = 1 if D <= cutoff else 0
	return kernel if lowpass else 1-kernel

def Filter(img, filter_matrix):
	shifted_dft_img = fftshift(fft2(img))
	filter_img = shifted_dft_img * filter_matrix
	return ifft2(ifftshift(filter_img))

for img in img_list:
	img1 = cv2.imread('./hw2_data/task1and2_hybrid_pyramid/{}'.format(img[0]), 0)
	img2 = cv2.imread('./hw2_data/task1and2_hybrid_pyramid/{}'.format(img[1]), 0)
	if img1.shape != img2.shape:
		if img1.shape > img2.shape:
			img2 = cv2.resize(img2, (img1.shape[1], img.shape[0]))
		else:
			img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

	gaussian_highpass = GaussianFilter(img1.shape[0], img1.shape[1], 30, False)
	gaussian_lowpass = GaussianFilter(img2.shape[0], img2.shape[1], 10, True)

	ideal_highpass = IdealFilter(img1.shape[0], img1.shape[1], 30, False)
	ideal_lowpass = IdealFilter(img2.shape[0], img2.shape[1], 10, True)

	high_f_img = Filter(img1, gaussian_highpass)
	low_f_img = Filter(img2, gaussian_lowpass)

	ideal_high_f_img = Filter(img1, ideal_highpass)
	ideal_low_f_img = Filter(img2, ideal_lowpass)

	merged_img = high_f_img + low_f_img
	output_img = np.real(merged_img)

	ideal_merged_img = ideal_high_f_img + ideal_low_f_img
	ideal_output_img = np.real(ideal_merged_img)

	cv2.imwrite("./hw2_data/task1and2_hybrid_pyramid/{}.png".format(img[0].split('_')[0] + '_output'), output_img)
	cv2.imwrite("./hw2_data/task1and2_hybrid_pyramid/{}.png".format(img[0].split('_')[0] + '_ideal_output'), ideal_output_img)