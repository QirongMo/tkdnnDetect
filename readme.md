# 代码说明
源代码是tkdnn，github地址为https://github.com/ceccocats/tkDNN.git，本代码主要是利用其darknet框架转tensorrt部署写了一个python接口。
# 使用说明
## 编译darknet导出各层的权重
先编译tkdnn推荐的darknet模型权重提取的代码https://git.hipert.unimore.it/fgatti/darknet.git，代码在本目录下的darknetTKDNN，编译时不要使用cuda和cudnn。
```bash
cd darknetTKDNN
make -j
```
然后，将模型各层的权重导出到自定义的“layers_dir”文件夹中，debug输出默认保留在本进程运行目录下的“debug”文件夹中，如果没有该文件夹需要先创建。
```bash
mkdir debug
./darknet export ${cfg_path} ${weights_path} ${layers_dir}
```

## tkdnn代码
### 代码修改说明
修改的主要代码都在darknetrt文件下，universal.cpp是用来将darknet模型转换未tensorrt模型的，而darknetRT.cpp则是python调用tensorrt进行推理的接口，test_darknetrt.py则是测试该python接口的。
### 使用方法
编译项目
```bash
cd tkdnn
mkdir build
cd build
cmake ..
make -j
```
转换权重, 保存tensorrt模型到“RT_file”
```bash
./universal ${layers_dir} ${debug_dir} ${cfg_path} ${names_file} ${RT_file}
```
### 注意事项

1、 模型输出的tensorrt模型未FP32，要输出FP16，则需要执行
```bash
export TKDNN_MODE=FP16
```
如果需要输出INT8，则还需要添加纠正数据，具体看tkdnn的readme
<br />
2、 文件格式需要注意
<br />
cfg文件和name文件的行尾序列要为LF，不要是CRLF。另外，cfg文件的内容尽量将空格删除。
<br />
3、找不到tensorrt
<br />
可在cmake/FindCUDNN.cmake中在if(NVINFER_LIBRARY)前添加tensortt的具体路径，如：
```cmake
set(NVINFER_LIBRARY /root/TensorRT-8.4.1.5/lib/libnvinfer.so)
set(NVINFER_INCLUDE_DIR /root/TensorRT-8.4.1.5/include)
if(NVINFER_LIBRARY)
```



