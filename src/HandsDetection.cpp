/*
authored by Joseph Augustus Fiume
*/

#include <opencv2/opencv.hpp>
#include <fstream>
#include <cassert>
#include <numeric>


using namespace cv;
using namespace std;
using namespace cv::dnn;


const int Y_THRESHOLD2 = 10;
const float NO_MATCH_THRESHOLD = 0.1;


namespace find_hands2 {

    void saveImage(string filePath, string fileName, const Mat &image, string fileType)
    {
        cv::imwrite(filePath + "/" + fileName + fileType, image);
    }



    bool compare(const Rect& a, const Rect& b)
    {
        if (abs(a.y - b.y) < Y_THRESHOLD2)
        {
            return a.x < b.x;
        }
        else
        {
            return a.y < b.y;
        }
    }



    double calculateIntersectionOverUnion(const Rect& rectA, const Rect& rectB)
    {
        Box boxA;
        boxA.x1 = rectA.x;
        boxA.x2 = rectA.x + rectA.width;
        boxA.y1 = rectA.y;
        boxA.y2 = rectA.y + rectA.height;
        Box boxB;
        boxB.x1 = rectB.x;
        boxB.x2 = rectB.x + rectB.width;
        boxB.y1 = rectB.y;
        boxB.y2 = rectB.y + rectB.height;
        // determine the (x, y)-coordinates of the intersection rectangle
        int xA = max(boxA.x1, boxB.x1);
        int yA = max(boxA.y1, boxB.y1);
        int xB = min(boxA.x2, boxB.x2);
        int yB = min(boxA.y2, boxB.y2);
        // compute the area of intersection rectangle
        double interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1);
        // compute the area of both the prediction and ground-truth
        // rectangles
        double boxAArea = (boxA.x2 - boxA.x1 + 1) * (boxA.y2 - boxA.y1 + 1);
        double boxBArea = (boxB.x2 - boxB.x1 + 1) * (boxB.y2 - boxB.y1 + 1);
        // compute the intersection over union by taking the intersection
        // area and dividing it by the sum of prediction + ground-truth
        // areas - the interesection area
        return (interArea / (boxAArea + boxBArea - interArea));
    }



    vector<Rect> readTestBoxes(string inFile)
    {
        vector<Rect> testRect;
        ifstream infile;
        infile.open(inFile);
        assert(infile);
        int x;
        int y;
        int width;
        int height;
        while (infile >> x >> y >> width >> height)
        {
            testRect.push_back(Rect(x, y, width, height));
        }
        sort(testRect.begin(), testRect.end(), compare);
        infile.close();
        return testRect;
    }



    void evaulateMissingHands(vector<Rect>& boxesA, const vector<Rect>& boxesB)
    {
        for (int i = 0; i < boxesA.size(); i++)
        {
            // missing boxes are somewhere in the middle of the vector
            double iou = calculateIntersectionOverUnion(boxesA[i], boxesB[i]);
            if (iou < NO_MATCH_THRESHOLD)
            {
                // add an empty box to the vector
                auto itPos1 = boxesA.begin() + i;
                auto newIt1 = boxesA.insert(itPos1, Rect(0, 0, 0, 0));
            }
        }
        // missing boxes are at the end of the vector
        if (boxesB.size() == boxesA.size() + 1)
        {
            // add an empty box to the vector
            auto itPos2 = boxesA.end();
            auto newIt2 = boxesA.insert(itPos2, Rect(0, 0, 0, 0));
        }
    }



    void addGroundTruthBoxes(Mat& inputImage, const vector<Rect>& groundTruthBoxes)
    {
        for (int i = 0; i < groundTruthBoxes.size(); i++)
        {
            rectangle(inputImage, groundTruthBoxes[i], RED, THICKNESS);
        }
    }



    // Performace Measurement Intersection over Union for hand detection
    void printIntersectionOverUnion(vector<Hands>& allBoxes,
                                    const vector<Hands>& allTestBoxes)
    {
        vector<double> ious;
        for (int i = 0; i < allTestBoxes.size(); i++)
        {
            cout << "name: " << allTestBoxes[i].name << endl;
            // find corresponding images to the tests
            int index = 0;
            while (allBoxes[index].name != allTestBoxes[i].name)
            {
                index++;
            }
            if (allBoxes[index].box.size() != allTestBoxes[i].box.size())
            {
                evaulateMissingHands(allBoxes[index].box, allTestBoxes[i].box);
            }
            for (int j = 0; j < allTestBoxes[i].box.size(); j++)
            {
                double iou = calculateIntersectionOverUnion(allBoxes[index].box[j],
                                                            allTestBoxes[i].box[j]);
                
                cout << iou << endl;
                ious.push_back(iou);
            }
        }
        double average = accumulate(ious.begin(), ious.end(), 0.0) / ious.size();
        cout << "average intersection over union of all hands: " << average << endl;
    }
}
