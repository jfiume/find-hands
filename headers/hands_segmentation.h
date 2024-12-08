/**
@author Alberto Ursino

@date 29/07/2022
*/

#ifndef HANDS_SEGMENTATION
#define HANDS_SEGMENTATION

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat segmentHand(Mat image);

Mat drawSegHands(Mat image, vector<Rect> b_boxes, Mat cropped_hand_seg, int index);
#include "../src/hands_segmentation.cpp"
#endif
