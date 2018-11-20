/* This code will generate a fractal image. Uses OpenCV, to compile:
   nvcc CudaFinal.cu `pkg-config --cflags --libs opencv`  */
#include <stdio.h>
#include <stdlib.h>
#include <opencv/highgui.h>
#include "utils/cheader.h"


typedef enum color {BLUE, GREEN, RED} Color;

__global__ void convert_to_hsv(unsigned char *src, float *hsv, int width, int heigth, int step, int channels) {
	float r, g, b;
	float h, s, v;
	int ren,col;

	ren = blockIdx.x;
	col = threadIdx.x;

	r = src[(ren * step) + (col * channels) + RED] / 255.0f;
	g = src[(ren * step) + (col * channels) + GREEN] / 255.0f;
	b = src[(ren * step) + (col * channels) + BLUE] / 255.0f;
	
	float max = fmax(r, fmax(g, b));
	float min = fmin(r, fmin(g, b));
	float diff = max - min;
	
	v = max;
	
	if(v == 0.0f) { // black
		h = s = 0.0f;
	} else {
		s = diff / v;
		if(diff < 0.001f) { // grey
			h = 0.0f;
		} else { // color
			if(max == r) {
				h = 60.0f * (g - b)/diff;
				if(h < 0.0f) { h += 360.0f; }
			} else if(max == g) {
				h = 60.0f * (2 + (b - r)/diff);
			} else {
				h = 60.0f * (4 + (r - g)/diff);
			}
		}		
	}
	// confusion line
	float minh=40.0f;
	float maxh=200.0f;
	float minis = 0;
	float maxs = 100;
	float miniv = 0;
	float maxv = 100;
		
	// if conditionals to check the color blindness line, if the pixel is in this line i change the color to other color base shifting the h	
	if (h > minh && h < maxh && s > minis && s < maxs && v > miniv && v < maxv){
		
		hsv[(ren * step) + (col * channels) + RED] =  (float) (h + 140.0f);
		hsv[(ren * step) + (col * channels) + GREEN] = (float) (s);
		hsv[(ren * step) + (col * channels) + BLUE] = (float) (v);
	} else { // this keep the pixel if it is out of the color blindnessline
		hsv[(ren * step) + (col * channels) + RED] =  (float) (h);
		hsv[(ren * step) + (col * channels) + GREEN] = (float) (s);
		hsv[(ren * step) + (col * channels) + BLUE] = (float) (v);
	}
	
	
}

__global__ void convert_to_rgb(float *hsv, unsigned char *dest, int width, int heigth, int step, int channels) {
	float r, g, b;
	float h, s, v;
	int ren,col;

	ren = blockIdx.x;
	col = threadIdx.x;	
	h = hsv[(ren * step) + (col * channels) + RED];
	s = hsv[(ren * step) + (col * channels) + GREEN];
	v = hsv[(ren * step) + (col * channels) + BLUE];
	
	float f = h/60.0f;
	float hi = floorf(f);
	f = f - hi;
	float p = v * (1 - s);
	float q = v * (1 - s * f);
	float t = v * (1 - s * (1 - f));
	
	if(hi == 0.0f || hi == 6.0f) {
		r = v;
		g = t;
		b = p;
	} else if(hi == 1.0f) {
		r = q;
		g = v;
		b = p;
	} else if(hi == 2.0f) {
		r = p;
		g = v;
		b = t;
	} else if(hi == 3.0f) {
		r = p;
		g = q;
		b = v;
	} else if(hi == 4.0f) {
		r = t;
		g = p;
		b = v;
	} else {
		r = v;
		g = p;
		b = q;
	}

	dest[(ren * step) + (col * channels) + RED] =  (unsigned char) __float2uint_rn(255.0f * r);
	dest[(ren * step) + (col * channels) + GREEN] = (unsigned char) __float2uint_rn(255.0f * g);
	dest[(ren * step) + (col * channels) + BLUE] = (unsigned char) __float2uint_rn(255.0f * b);
}


int main(int argc, char* argv[]) {
	int size, step, size2;
	int i;
	double acum; 
	float *dev_hsv;
	unsigned char *dev_src ,*dev_dest;
	
		
	if (argc != 2) {
		printf("usage: %s source_file\n", argv[0]);
		return -1;
	}
	// creation of the matrixes using OpenCV
	IplImage *src = cvLoadImage(argv[1], CV_LOAD_IMAGE_COLOR);
	IplImage *hsv = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_32F, 3);
	IplImage *dest = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, 3);
	
	// if the source is incorrect
	if (!src) {
		printf("Could not load image file: %s\n", argv[1]);
		return -1;
	}
	
	// calculate the size for the RGB matrixes and the HSV matrix
	size = src->width * src->height * src->nChannels * sizeof(uchar);
	size2 = src->width * src->height * src->nChannels * sizeof(float);

	// Allocate the memory for the matrixes in the GPU
	cudaMalloc((void**) &dev_src, size);
	cudaMalloc((void**) &dev_hsv, size2);
	cudaMalloc((void**) &dev_dest, size);
	
	// Copy the Matrix from the source image to the GPU
	cudaMemcpy(dev_src, src->imageData, size, cudaMemcpyHostToDevice);
	
	//initialize the timer
	acum = 0;
	step = src->widthStep / sizeof(uchar);

	printf("Starting...\n");
	for (i = 0; i < N; i++) {
		// Start The timer
		start_timer();
		// Call the kernel to convert from RGB to HSV
		convert_to_hsv<<<src->height, src->width>>>(dev_src, dev_hsv, src->width, src->height, step, src->nChannels);
		// copy the matrix to the Host matrix for hsv
		cudaMemcpy(hsv->imageData, dev_hsv, size2, cudaMemcpyDeviceToHost);
		// Call the kernel to convert from HSV to RGB
		convert_to_rgb<<<src->height, src->width>>>(dev_hsv, dev_dest, src->width, src->height, step, src->nChannels);
		// copy the computed matrix in RGB to the host to be printed
		cudaMemcpy(dest->imageData, dev_dest, size, cudaMemcpyDeviceToHost);
		// add the time to the timer
		acum += stop_timer();
	}
	
	// free the memory
	cudaFree(dev_dest);
	cudaFree(dev_hsv);
	cudaFree(dev_src);
	
	printf("avg time = %.5lf ms\n", (acum / N));
	
	cvShowImage("Image (Original)", src);
	cvShowImage("Image (Computed)", dest);
	cvWaitKey(0);
	cvDestroyWindow("Image (Original)");
	cvDestroyWindow("Image (Computed)");

	return 0;
}
