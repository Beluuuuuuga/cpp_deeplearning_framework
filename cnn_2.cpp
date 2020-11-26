#include <iostream>
#include <stdlib.h> /*rand関数を使う宣言*/
#include <time.h> /*time関数の使用宣言*/
#include <fstream>

#include "methods.h"
using namespace std;

int main(){
    int i, n;
    FILE *fp;

    FILE *w0_fp;
    FILE *w1_fp;
    FILE *w2_fp;
    FILE *w3_fp;
    
    FILE *b0_fp;
    FILE *b1_fp;
    FILE *b2_fp;
    FILE *b3_fp;

    /*  読み込みモードでファイルをオープン  */
    // mnistの画像
    // fp = fopen("data/sample_mnist_1-3_float.txt", "r");
    // fp = fopen("data/sample_1.txt", "r");
    fp = fopen("data/sample_1_2.txt", "r");
    
    // weights
    // w0_fp = fopen("data/w0.txt", "r");
    // w1_fp = fopen("data/w1.txt", "r");
    // w2_fp = fopen("data/w2.txt", "r");
    // w3_fp = fopen("data/w3.txt", "r");

    // weights_2
    w0_fp = fopen("data/w0_2.txt", "r");
    w1_fp = fopen("data/w1_2.txt", "r");
    w2_fp = fopen("data/w2_2.txt", "r");
    w3_fp = fopen("data/w3_2.txt", "r");

    // bias
    b0_fp = fopen("data/b0.txt", "r");
    b1_fp = fopen("data/b1.txt", "r");
    b2_fp = fopen("data/b2.txt", "r");
    b3_fp = fopen("data/b3.txt", "r");

    // 変数作成
    static const int kWidths[] = {28, 14, 7};
    static const int kHeights[] = {28, 14, 7};
    static const int kChannels[] = {1, 16, 32, 1024 ,10};
    float x1[kWidths[0] * kHeights[0] * kChannels[1]]; // 出力
    float x2[kWidths[0] * kHeights[0] * kChannels[1]]; // 出力
    float x3[kWidths[1] * kHeights[1] * kChannels[1]]; // 出力
    float x4[kWidths[1] * kHeights[1] * kChannels[2]]; // 出力
    float x5[kWidths[1] * kHeights[1] * kChannels[2]]; // 出力
    float x6[kWidths[2] * kHeights[2] * kChannels[2]]; // 出力
    float x7[kChannels[3]]; // 出力
    float x8[kChannels[3]]; // 出力
    float y[10]; // 出力
    float y2[10]; // 出力

    float x[1*28*28]; // input image data

    // weight
    float weight0[16*1*3*3]; // 144, conv_1
    float weight1[32*16*3*3]; // 4608, conv_2
    float weight2[1024*32*7*7]; // 1605632, fcc
    float weight3[1024*10]; // 10240, output

    // bias
    float bias0[16]; // 16, conv_1
    float bias1[32]; // 32, conv_2
    float bias2[1024]; // 1024, fcc
    float bias3[10]; // 10, output

    // mnist画像読み込み
    for(i=0; i < 28*28; i++){
        fscanf(fp, "%f", &(x[i]));
    }
    // for(i=28*28; i < 28*28*2; i++){
    //     fscanf(fp, "%f", &(x[i]));
    // }

    // weight読み込み
    for(i=0; i < 16*1*3*3; i++){
        fscanf(w0_fp, "%f", &(weight0[i]));
    }
    for(i=0; i < 32*16*3*3; i++){
        fscanf(w1_fp, "%f", &(weight1[i]));
    }
    for(i=0; i < 1024*32*7*7; i++){
        fscanf(w2_fp, "%f", &(weight2[i]));
    }
    for(i=0; i < 1024*10; i++){
        fscanf(w3_fp, "%f", &(weight2[i]));
    }

    // bias読み込み
    for(i=0; i < 16; i++){
        fscanf(b0_fp, "%f", &(bias0[i]));
    }
    for(i=0; i < 32; i++){
        fscanf(b1_fp, "%f", &(bias1[i]));
    }
    for(i=0; i < 1024; i++){
        fscanf(b2_fp, "%f", &(bias2[i]));
    }
    for(i=0; i < 10; i++){
        fscanf(b3_fp, "%f", &(bias3[i]));
    }
 
    fclose(fp);

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
    
    // for (int i=0; i<10; ++i){
    //     cout << y[i] << endl;
    // }

    cout << endl;
    softmax(y, y2);
    for (int i=0; i<10; ++i){
        cout << y2[i] << endl;
    }
    // cout << endl;
    // for (int i=0; i<28*28; ++i){
    //     // cout <<  << endl;
    //     cout << x[i] << endl;
    // }

    return 0;

}