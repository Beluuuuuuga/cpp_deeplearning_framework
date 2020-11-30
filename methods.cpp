
#include "methods.h"

void linear(const float *x, const float* weight, const float* bias,
            int in_features, int out_features, float *y) {
  for (int i = 0; i < out_features; ++i) {
    float sum = 0.f;
    for (int j = 0; j < in_features; ++j) {
      sum += x[j] * weight[i * in_features + j];
    }
    y[i] = sum + bias[i];
  }
}

void relu(const float *x, int size, float *y) {
  for (int i = 0; i < size; ++i) {
    y[i] = std::max(x[i], .0f);
  }
}

void softmax(float *y, float *y2){
  float output_sum = 0;
  for (int i = 0; i < 10; ++i){
    y2[i] = exp(y[i]);
    output_sum += y2[i];
  }
  for (int i = 0; i < 10; ++i){
    y2[i] = y2[i]/output_sum;
  }
}


void maxpooling(const float *x, int width, int height, int channels, int stride, float *y) {
  for (int ch = 0; ch < channels; ++ch) {
    for (int h = 0; h < height; h += stride) {
      for (int w = 0; w < width; w += stride) {
        float maxval = -INT8_MAX;

        for (int bh = 0; bh < stride; ++bh) {
          for (int bw = 0; bw < stride; ++bw) {
            maxval = std::max(maxval, x[(ch * height + h + bh) * width + w + bw]);
          }
        }

        y[(ch * (height / stride) + (h / stride)) * (width / stride) + w / stride] = maxval;
      }
    }
  }
}


float calcu_convolution(const float* x, const float* weight, int height, int width, int filter_n, int h, int w,int input_channels, int ksize, float sum){
    for (int ich = 0; ich < input_channels; ++ich) {
        for (int kh = 0; kh < ksize; ++kh) {
            for (int kw = 0; kw < ksize; ++kw) {

                // offsetを計算 ksize3/2->1, ksize5/2->2
                int ph = h + kh - ksize/2;
                int pw = w + kw - ksize/2;

                // 画像はみ出さないように調整している
                if (ph < 0 || ph >= height || pw < 0 || pw >= width) {
                    continue;
                }

                int pix_idx = (ich * height + ph) * width + pw;
                // 多次元を1次元に変換して計算している
                int weight_idx = ((filter_n * input_channels + ich) * ksize + kh) * ksize + kw;

                sum += x[pix_idx] * weight[weight_idx];
            }
        }
    }
    return sum;
}

void convolution(const float* x, const float* weight, const float* bias, int width, int height,
            int input_channels, int filtersize, int ksize, float* y) {
    for (int filter_n = 0; filter_n < filtersize; ++filter_n) {
        // 今回はゼロパディングを予定してるので幅・高さは変更なし
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                float sum = 0.f;

                sum = calcu_convolution(x, weight, height, width, filter_n, h, w, input_channels, ksize, sum);
                
                sum += bias[filter_n];
                y[(filter_n * height + h) * width + w] = sum;

            }
        }
    }
}
