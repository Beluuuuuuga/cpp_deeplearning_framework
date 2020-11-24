#include <iostream>
#include <stdlib.h> /*rand関数を使う宣言*/
#include <time.h> /*time関数の使用宣言*/
#include <fstream>

#include "methods.h"
using namespace std;

int main(){
    int i, n;
    FILE *fp;
 
    // mnistの画像
    fp = fopen("data/sample_mnist_1-3_float.txt", "r");     /*  読み込みモードでファイルをオープン  */

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
    float weight0[4*1*3*3];
    float bias0[4] = {10,10,10,10};
    float weight1[8*4*3*3] = {};
    float bias1[8] = {};
    float weight2[32*8*7*7] = {};
    float bias2[8] = {};
    float weight3[10*32] = {};
    float bias3[10] = {};

    // srand((unsigned int)time(NULL)); /*乱数の初期化*/
    // srand(time(NULL));

    // mnist画像読み込み
    for(i=0; i < 28*28; i++){
        fscanf(fp, "%f", &(x[i]));
    }
 
    fclose(fp);

    for(int i = 0; i < 4*3*3 ; i++)
    {
        weight0[i] = i*100;
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
    return 0;

}