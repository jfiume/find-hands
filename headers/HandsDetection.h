/*
authored by Joseph Augustus Fiume
*/

#ifndef HANDSDETECTION_H
#define HANDSDETECTION_H

#include <opencv2/core/mat.hpp>
#include <vector>

namespace find_hands2 {
    struct Box {
        int x1;
        int y1;
        int x2;
        int y2;
    };
    struct Hands {
        std::string name;
        std::vector<cv::Rect> box;
        cv::Mat handImage;
    }; 
    void saveImage(std::string filePath, std::string fileName,
                    const cv::Mat &image, std::string fileType);
    bool compare(const cv::Rect& a, const cv::Rect& b);
    double calculateIntersectionOverUnion(const cv::Rect& boxA, const cv::Rect& boxB);
    std::vector<cv::Rect> readTestBoxes(const std::vector<std::string>& testFiles);
    void evaulateMissingHands(std::vector<cv::Rect>& boxesA,
                              const std::vector<cv::Rect>& boxesB);
    void addGroundTruthBoxes(cv::Mat& inputImage,
                             const std::vector<cv::Rect>& groundTruthBoxes);
    void printIntersectionOverUnion(std::vector<Hands>& allBoxes,
                                    const std::vector<Hands>& allTestBoxes);
}

#include "../src/HandsDetection.cpp"
#endif //HANDSDETECTION_H
