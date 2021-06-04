#ifndef DARKNETDETECTOR_H
#define DARKNETDETECTOR_H

#define Linux

#include <iostream>
#include <fstream>
#include <vector>
#include <string.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#ifdef WIN32
#include <yolo_v2_class.hpp>
#endif

#include "../thirdparty/darknet/include/darknet.h"

using namespace std;
using namespace cv;

struct BBox
{
    double x = 0.0;
    double y = 0.0;
    double w = 0.0;
    double h = 0.0;
    int label = 0;
    double score = 0.0;
};

struct Info
{
    int img_width_{0};
    int img_height_{0};
    int img_format_{0};
    int bbox_num_{0};

    int total_size_{0};
};

struct ConnectedData
{
    Info info;

    vector<int> data_;

    inline int getRed(int i, int j)
    {
        return data_[3 *(j * info.img_width_ + i)];
    }
    inline int getGreen(int i, int j)
    {
        return data_[3 * (j * info.img_width_ + i) + 1];
    }

    inline int getBlue(int i, int j)
    {
        return data_[3 * (j * info.img_width_ + i) + 2];
    }

    inline int getBBoxX(int i)
    {
        return getInt(5 * i);
    }

    inline int getBBoxY(int i)
    {
        return getInt(5 * i + 1);
    }

    inline int getBBoxW(int i)
    {
        return getInt(5 * i + 2);
    }

    inline int getBBoxH(int i)
    {
        return getInt(5 * i + 3);
    }

    inline int getBBoxLabel(int i)
    {
        return getInt(5 * i + 4);
    }

    inline void setPixel(int i,int j, int r, int g, int b)
    {
        data_[3 *(j * info.img_width_ + i)] = r;
        data_[3 *(j * info.img_width_ + i) + 1] = g;
        data_[3 *(j * info.img_width_ + i) + 2] = b;
    }

    inline void resetBBoxNum(int reset_bbox_num)
    {
        info.bbox_num_ = reset_bbox_num;
        data_.resize(info.bbox_num_ * 4 * 5);
    }

    inline void setBBox(int i, int x, int y, int w, int h, int label)
    {
        setInt(5 * i, x);
        setInt(5 * i + 1, y);
        setInt(5 * i + 2, w);
        setInt(5 * i + 3, h);
        setInt(5 * i + 4, label);
    }

    inline void setInt(int i, int data)
    {
        data_[4 * i] = (uchar)(0x000000ff & data);
        data_[4 * i + 1] = (uchar)((0x0000ff00 & data) >> 8);
        data_[4 * i + 2] = (uchar)((0x00ff0000 & data) >> 16);
        data_[4 * i + 3] = (uchar)((0xff000000 & data) >> 24);
    }

    inline int getInt(int i)
    {
        int data = data_[4 * i] & 0x000000ff;
        data |= ((data_[4 * i + 1] << 8) & 0x0000ff00);
        data |= ((data_[4 * i + 2] << 16) & 0x00ff0000);
        data |= ((data_[4 * i + 3] << 24) & 0xff000000);

        return data;
    }

    void clear()
    {
        info.img_width_ = 0;
        info.img_height_ = 0;
        info.img_format_ = 0;
        info.bbox_num_ = 0;

        info.total_size_ = 0;

        data_.clear();
    }

    vector<vector<int>> toBBox()
    {
        {
            vector<vector<int>> bbox_result;

            bbox_result.resize(info.bbox_num_);

            for(int i = 0; i < info.bbox_num_; ++i)
            {
                bbox_result[i].emplace_back(getBBoxX(i));
                bbox_result[i].emplace_back(getBBoxY(i));
                bbox_result[i].emplace_back(getBBoxW(i));
                bbox_result[i].emplace_back(getBBoxH(i));
                bbox_result[i].emplace_back(getBBoxLabel(i));
            }

            return bbox_result;
        }
    }

    void outputInfo();
};

class DarknetDetector
{
public:
    DarknetDetector(const std::string &yolov3_cfg, const std::string &yolov3_weights, const std::string &coco_data);

    ~DarknetDetector();

    std::vector<std::pair<char *, std::vector<float>>> getDarknetResult(image img, float thresh=0.5, float hier_thresh=0.5, float nms=0.45);

    std::vector<std::pair<char *, std::vector<float>>> getDarknetResult(char *img, float thresh=0.5, float hier_thresh=0.5, float nms=0.45);

    std::vector<std::pair<char *, std::vector<float>>> getDarknetResult(float *img, int w, int h, int c, float thresh=0.5, float hier_thresh=0.5, float nms=0.45);

private:
    std::vector<std::pair<char *, std::vector<float>>> detect(image im, float thresh=0.5, float hier_thresh=0.5, float nms=0.45);

    std::vector<std::pair<char *, std::vector<float>>> detect(char *img, float thresh=0.5, float hier_thresh=0.5, float nms=0.45);

    std::vector<std::pair<char *, std::vector<float>>> detect(float *img, int w, int h, int c, float thresh=0.5, float hier_thresh=0.5, float nms=0.45);

    bool darknet_process(ConnectedData &data);

    network *net;
    metadata meta;
};

#endif // DARKNETDETECTOR_H
