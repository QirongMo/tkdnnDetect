#include <stdlib.h>
#include <iostream>
#include <signal.h>
#include <mutex>
#include <stack>
#include "Yolo3Detection.h"
using namespace std;
using namespace tk::dnn;

typedef struct image {
    int w;
    int h;
    int c;
    float *data;
} image;


struct Box
{
    float confidence;
    float x1;
    float y1;
    float x2;
    float y2;
    int cls_id;
};

struct RT_Result
{
    int num_boxes;
    Box *boxes;
};

extern "C"{
    image make_image(int w, int h, int c);
    void copy_image_from_bytes(image img, char *pdata);
    void free_image(image img);

    Yolo3Detection *load_network(const char *weight_file, int classes, int batch_size, float thresh);
    RT_Result* detect_img(Yolo3Detection *net, image img);
}