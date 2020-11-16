#include <iostream>
using namespace std;


void calcu_convolution(const float* x, const float* weight, int height, int width, int filter_n, int h, int w,int input_channels, int ksize, float sum){
    for (int ich = 0; ich < input_channels; ++ich) {
        for (int kh = 0; kh < ksize; ++kh) {
            for (int32_t kw = 0; kw < ksize; ++kw) {

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
}

void convolution(const float* x, const float* weight, const float* bias, int width, int height,
            int input_channels, int filtersize, int ksize, float* y) {
    for (int filter_n = 0; filter_n < filtersize; ++filter_n) {
        // 今回はゼロパディングを予定してるので幅・高さは変更なし
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                float sum = 0.f;

                calcu_convolution(x, weight, height, width, filter_n, h, w, input_channels, ksize, sum);

            }
        }
    }
}


int main(){
    // 変数作成
    static const int kWidths[] = {28, 14, 7};
    static const int kHeights[] = {28, 14, 7};
    static const int kChannels[] = {1, 4, 8, 32, 10};
    float x1[kWidths[0] * kHeights[0] * kChannels[1]]; // 出力->y
    float x2[kWidths[0] * kHeights[0] * kChannels[1]];
    float x3[kWidths[1] * kHeights[1] * kChannels[1]];

    // float x[1*28]
    const float x[1*28*28] = {};
    const float weight0[4*1*3*3] = {};
    const float bias0[4] = {};

    convolution(x, weight0, bias0, kWidths[0], kHeights[0], kChannels[0], kChannels[1], 3, x1);
            
}