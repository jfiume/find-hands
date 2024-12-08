/*
authored by Joseph Augustus Fiume
*/

/*
 instructions to run from command line input in the "build" folder:
    cmake ../src
    make
    ./find_hands ../rgb/01.jpg
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include "../headers/hands_detection.h"
#include "../headers/HandsDetection.h"
#include "../headers/LoadDirectory.h"
#include "../headers/hands_segmentation.h"


using namespace cv;
using namespace std;
using namespace cv::dnn;
using namespace find_hands;
using namespace find_hands2;


const char* TEST_BOX_FILE_TYPE = ".txt";
const char* INPUT_FILE_TYPE = ".jpg";
const char* TEST_PIXELS_FILE_TYPE = "png";
// network path
const string ONNX = "../MLInput/best2.onnx";


int main(int argc, char** argv)
{
    if (argc != 2)
    {
        cout << "program requires 1 image" << endl;
        return -1;
    }   
    String imagePath = argv[1];
	Mat source;
	try {
		source = imread(imagePath);
	}
	catch (int e) {
		cout << "An exception occurred. Exception Nr. " << e << '\n';
	}
	//namedWindow("input image", WINDOW_AUTOSIZE);
	//imshow("input image", source);
    //
    // Load the network
    //
    // Load class list
    vector<string> class_list;
    class_list.push_back("hand");
    // Load model
    Net net;
    net = readNet(ONNX);
    // check that network was loaded
    if (net.empty())
    {
        cout << "load Network failed" <<endl;
        return -1;
    }
    //
    // hand detection
    //
    // add hand boxes to images
    Mat segmented_hands = source.clone();
    vector<Rect> detections = draw_bouding_boxes(source, net, class_list);
    Mat full_mask = Mat::zeros(Size(source.cols, source.rows), CV_8UC1);;
    for (int i = 0; i < detections.size(); i++)
    {
        // Doing segmentation on each cropped hand
        Mat cropped_hand = source(detections[i]);
        Mat cropped_hand_seg = segmentHand(cropped_hand);

        segmented_hands = drawSegHands(segmented_hands, detections, cropped_hand_seg, i);
    }
        
    // uncomment to save images
//        saveImage("../output_images", "boxes_" + imageBoxes.name, frame, ".jpg");
    
    // show images
    imshow("Segmented Hands Output", segmented_hands);
    waitKey(0);
    imshow("Hand Detection Output", source);
    waitKey(0);

    
    
    destroyAllWindows();
    return 0;
}
