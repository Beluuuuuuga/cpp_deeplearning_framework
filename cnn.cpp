#include <iostream>
using namespace std;


// kernelsizeで畳み込みを行う
// void calcu_convolution(int input_channel, int ksize, int x, int y, int height, int width){

//     double sum = 0; // totalの値

//     for (int input_c = 0; input_c < input_channel; ++ input_c){
//         for (int kh = 0; kh < ksize; ++kh){
//             for (int kh = 0; kh < ksize; ++kh){

//                 // offsetを計算 ksize3/2->1, ksize5/2->2
//                 int target_x = x + kw - (ksize/2) 
//                 int target_y = y + kh - (ksize/2)

//                 // 画像はみ出さないように調整している
//                 if (target_x < 0 || target_x >= height || target_y < 0 || target_y >= width) {
//                   continue;
//                 }


                
//             }
//         }
//     }
// }

// 画像をピクセル単位で計算
// fitersizeはoutputchannelになる
// void convolution(int filtersize, int height, int width, int input_channel){
//     for (int filter_n = 0; filter_n < filtersize; ++filter_n){
//         // 今回はゼロパディングを予定してるので幅・高さは変更なし
//         for (int h = 0; h < height; ++h){
//             for (int w = 0; w < width; ++w){
//                 cout << h << h/2 << endl;
//             }
//         }
//     }
//     // return 5;
// }

void convolution(const float* x, const float* weight, const float* bias, int width, int height,
            int in_channels, int out_channels, int ksize, float* y) {

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