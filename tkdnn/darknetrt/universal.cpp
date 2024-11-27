#include <iostream>
#include <vector>
#include <dirent.h>
#include <iomanip>
#include "tkdnn.h"
#include "test.h"
#include "DarknetParser.h"
#include<algorithm>

using namespace std;
const string YOLO_PARTTEN = "yolo";


int listdir(const char *dir_root, vector<string> *outputs)
{
    DIR *dir;
    struct dirent *diread;
    // if ((dir = opendir("./")) != nullptr)
    if ((dir = opendir(dir_root)) != nullptr)
    {
        while ((diread = readdir(dir)) != nullptr)
        {
            outputs->push_back(string(diread->d_name));
        }
        closedir(dir);
    }
    else
    {
        perror("opendir");
        return EXIT_FAILURE;
    }
    return 0;
}

int get_out_bin(const char *debug_dir, const char *layers_dir, vector<string> *outputs)
{
    DIR *dir;
    struct dirent *diread;
    if ((dir = opendir(layers_dir)) != nullptr)
    {
        while ((diread = readdir(dir)) != nullptr)
        {
            string layer_filename = string(diread->d_name);
            size_t start = layer_filename.find("g");
            size_t end  = layer_filename.find(".bin");
            if(start == 0 && end != std::string::npos){
                string debug_filename = "layer"+layer_filename.substr(start+1, end-start-1)+"_out.bin";
                outputs->push_back(string(debug_dir)+debug_filename);
            }
        }
        closedir(dir);
        sort((*outputs).begin(),(*outputs).end());
    }
    else
    {
        perror("opendir");
        return EXIT_FAILURE;
    }
    return 0;
}

int main(int argc, char *argv[])
{
    string root_dir;

    for (int i = 0; i < argc; ++i)
    {
        cout << "arg: " << setfill(' ') << setw(2) << i << "\t" << argv[i] << endl;
    }
    if (argc != 6)
    {
        cout << "try: './universal [layers_dir] [debug_dir] [cfg_file] [names_file] [RT_file]'" << endl;
        return EXIT_FAILURE;
    }
    // 1. Find the path of input and output layer.
    vector<string> input_bins = {string(argv[1]) + "input.bin"};
    // vector<string> debug_outs;
    // int flag = listdir(argv[2], &debug_outs);
    // if (flag != 0)
    // {
    //     return flag;
    // }
    // // vector<string> output_bins;
    // // for (string tfile : debug_outs)
    // // {
    // //     if (tfile.find(YOLO_PARTTEN) != string::npos)
    // //     {
    // //         cout<<string(argv[2]) + tfile<<endl;
    // //         output_bins.push_back(string(argv[2]) + tfile);
    // //     }
    // // }
    // vector<string> output_bins = {
    //     string(argv[2]) + "layer144_out.bin",
    //     string(argv[2]) + "layer159_out.bin",
    //     string(argv[2]) + "layer174_out.bin"
    // };
    vector<string> output_bins;
    int flag = get_out_bin(argv[2], argv[1], &output_bins);
    if (flag != 0)
    {
        return flag;
    }
    // 2. Parse network.
    tk::dnn::Network *net =tk::dnn::darknetParser(argv[3],argv[1],argv[4]);
    // 3. Convert to TensorRT
    tk::dnn::NetworkRT *netRT= new tk::dnn::NetworkRT(net,argv[5]);
    // 4. Test.
    int ret=testInference(input_bins,output_bins,net,netRT);
    cout<<"Test output: "<<ret<<endl;
    // 5. Clean.
    // net->releaseLayers();
    delete net;
    netRT->destroy();
    delete netRT;
    cout<<"ret:"<<ret<<endl;
    return ret;
}
