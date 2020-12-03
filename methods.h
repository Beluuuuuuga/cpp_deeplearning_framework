
#include <stdlib.h> /*rand関数を使う宣言*/
#include <iostream>
#include <math.h>

void linear(const float *x, const float* weight, const float* bias,
            int in_features, int out_features, float *y);

void relu(const float *x, int size, float *y);

void softmax(float *y, float *y2);

void maxpooling(const float *x, int width, int height, int channels, int stride, float *y);

float calcu_convolution(const float* x, const float* weight, int height, int width, int filter_n, int h, int w,int input_channels, int ksize, float sum);

void convolution(const float* x, const float* weight, const float* bias, int width, int height,
            int input_channels, int filtersize, int ksize, float* y);

void argmax(float *activated_y, int &max_index);