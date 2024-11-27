#include "darknetRT.h"

image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

image make_image(int w, int h, int c){
    image out = make_empty_image(w,h,c);
    out.data = (float*)calloc(h * w * c, sizeof(float));
    return out;
}

void copy_image_from_bytes(image img, char *pdata){
    int w = img.w;
    int h = img.h;
    int c = img.c;
    int i, k, j;

    for (k = 0; k < c; ++k) {
        for (j = 0; j < h; ++j) {
            for (i = 0; i < w; ++i) {
                int index = k + c * i + c * w*j;
                float val = pdata[index];
                img.data[index] = val;
            }
        }
    }
}

cv::Mat convert_image_to_mat(image img){
    cv::Mat mat = cv::Mat(img.h, img.w, CV_8UC(img.c));
    int i, k, j;
    for (k = 0; k < img.c; ++k) {
        for (j = 0; j < img.h; ++j) {
            for (i = 0; i < img.w; ++i) {
                int index = k + img.c * i + img.c * img.w*j;
                float val = img.data[index];
                mat.data[index] = (unsigned char)val;
            }
        }
    }
    return mat;
}

void free_image(image img){ 
    if(img.data){
        free(img.data);
    }
}


// Load network (get instance)
Yolo3Detection *load_network(const char *weight_file, int classes, int batch_size, float thresh)
{
    Yolo3Detection *net = new Yolo3Detection();
    net->init(weight_file, classes, batch_size, thresh);
    return net;
}

void clean_data(Yolo3Detection *net){
    net->batchDetected.clear();
    net->detected.clear();
    net->stats.clear();
}

RT_Result* detect_img(Yolo3Detection *net, image img){
    vector<cv::Mat> frames;
    cv::Mat frame = convert_image_to_mat(img);
    // cv::imwrite("org.jpg", frame);
    frames.push_back(frame);
    // int size = frames.size();
    int size = 1;
    RT_Result *results = (RT_Result*)calloc(size, sizeof(RT_Result));
    net->update(frames, size);
    for(int i=0; i<size; ++i){
        vector<tk::dnn::box> bbox = net->batchDetected[i];
        int num_boxes = bbox.size();
        Box* boxes = (Box*)calloc(num_boxes, sizeof(Box));
        for (int j = 0; j<num_boxes; ++j){
            Box rts;
            rts.confidence = bbox[j].prob;
            rts.cls_id = bbox[j].cl;
            rts.x1 = bbox[j].x;
            rts.y1 = bbox[j].y;
            rts.x2 = bbox[j].x + bbox[j].w;
            rts.y2 = bbox[j].y + bbox[j].h;
            boxes[j] = rts;
            // cout<<rts.x1<<" "<<rts.x1<<" "<<rts.y1<<" "<<rts.x2<<" "<<rts.y2<<" "<<rts.confidence<<" "<<rts.cls_id<<endl;
            // cout<<bbox[j].x<<" "<<bbox[j].y<<" "<<bbox[j].w<<" "<<bbox[j].h<<" "<<bbox[j].prob<<" "<<bbox[j].cl<<endl;
        }
        results[i].num_boxes = num_boxes;
        results[i].boxes = boxes;
    }
    clean_data(net);
    frames.clear();
    return results;
}