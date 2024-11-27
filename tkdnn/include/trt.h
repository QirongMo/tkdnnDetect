#include <iostream>
#include <signal.h>
#include <stdlib.h>
#include <mutex>
#include <stack>
#include "Yolo3Detection.h"

using namespace std;
using namespace tk::dnn;


struct RT_Result
{
    float confidence;
    float x1;
    float y1;
    float x2;
    float y2;
    int cls_id;
};


// #ifdef __cplusplus
extern "C"
{
// #endif __cplusplus
    cv::Mat *image_to_mat(char *pdata, int w, int h, int c);
    Yolo3Detection *load_network(const char *weight_file, int classes, int batch_size, float thresh);
    RT_Result* detect_img(Yolo3Detection *net, cv::Mat frame, int *num_boxes);
    void free_image(cv::Mat *frame);

// #ifdef __cplusplus
}
// #endif __cplusplus