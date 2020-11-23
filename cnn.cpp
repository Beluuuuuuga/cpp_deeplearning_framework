#include <iostream>
#include <stdlib.h> /*rand関数を使う宣言*/
#include <time.h> /*time関数の使用宣言*/
using namespace std;


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


int main(){
    // 変数作成
    static const int kWidths[] = {28, 14, 7};
    static const int kHeights[] = {28, 14, 7};
    static const int kChannels[] = {1, 4, 8, 32, 10};
    float x1[kWidths[0] * kHeights[0] * kChannels[1]]; // 出力
    float x2[kWidths[0] * kHeights[0] * kChannels[1]]; // 出力
    float x3[kWidths[1] * kHeights[1] * kChannels[1]]; // 出力
    float x4[kWidths[1] * kHeights[1] * kChannels[2]]; // 出力
    float x5[kWidths[1] * kHeights[1] * kChannels[2]]; // 出力
    float x6[kWidths[2] * kHeights[2] * kChannels[2]]; // 出力
    float x7[kChannels[3]]; // 出力
    float x8[kChannels[3]]; // 出力
    float y[10]; // 出力

    // float x[1*28]
    // const float x[1*28*28] = {};
    float x[1*28*28];
    // float x[1][28][28];
    // const float weight0[4*1*3*3] = {1};
    float weight0[4*1*3*3];
    // float weight0[4][1][3][3];
    const float bias0[4] = {10,10,10,10};
    const float weight1[8*4*3*3] = {};
    const float bias1[8] = {};
    const float weight2[32*8*7*7] = {};
    const float bias2[8] = {};
    const float weight3[10*32] = {};
    const float bias3[10] = {};

    // srand((unsigned int)time(NULL)); /*乱数の初期化*/
    // srand(time(NULL));

    for(int i = 0; i < 28*28 ; i++)
    {
        x[i] = i;
    }
    for(int i = 0; i < 4*3*3 ; i++)
    {
        weight0[i] = i;
    }

    // 1
    convolution(x, weight0, bias0, kWidths[0], kHeights[0], kChannels[0], kChannels[1], 3, x1);
    relu(x1, kWidths[0] * kHeights[0] * kChannels[1], x2);
    maxpooling(x2, kWidths[0], kHeights[0], kChannels[1], 2, x3); // stride=>2

    // 2
    convolution(x3, weight1, bias1, kWidths[1], kHeights[1], kChannels[1], kChannels[1], 3, x4);
    relu(x4, kWidths[1] * kHeights[1] * kChannels[2], x5);
    maxpooling(x5, kWidths[1], kHeights[1], kChannels[2], 2, x6); // stride=>2

    // 3
    linear(x6, weight2, bias2, kWidths[2] * kHeights[2] * kChannels[2], kChannels[3], x7);
    relu(x7, kChannels[3], x8);

    // 4
    linear(x8, weight3, bias3, kChannels[3], kChannels[4], y);
    
    for (int i=0; i<14*14*4; ++i){
        cout << y[i] << endl;
    }

}