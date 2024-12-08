/*
authored by Fabio Marangoni
*/

#ifndef HANDS_DETECTION_H
#define HANDS_DETECTION_H

#include <opencv2/core/mat.hpp>
#include <vector>

namespace find_hands {
    std::vector<cv::Mat> pre_process(cv::Mat& input_image, cv::dnn::Net& net);
    void draw_label(cv::Mat& input_image, std::string label, int left, int top);
    std::vector<cv::Rect> post_process(cv::Mat& input_image,
                                  std::vector<cv::Mat>& outputs,
                                  const std::vector<std::string>& class_name);
    std::vector<cv::Rect> draw_bouding_boxes(cv::Mat& frame, cv::dnn::Net& net,
                                        const std::vector<std::string>& class_list);   
    bool compare(const cv::Rect& a, const cv::Rect& b);
}

#include "../src/hands_detection.cpp"
#endif //HANDS_DETECTION_H
