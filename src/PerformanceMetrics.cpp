/*
authored by Joseph Augustus Fiume
*/

/*
 instructions to run from command line input in the "build" folder:
    cmake ../src
    make
    ./find_hands ../rgb ../det
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


int performanceMetrics(int argc, char** argv)
{
    if (argc != 3)
    {
        cout << "program requires 3 directories" << endl;
        return -1;
    }   
    //
    // load directory of input images from command line
    //
    vector<string> files;
    int load_images = loadDirectory(files, argv[1], argv[1], INPUT_FILE_TYPE);
    // check that files were loaded
    if (load_images == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }
    // create a vector of mat images
    vector<Mat> images;
//    vector<Mat> originalImages;
    for (size_t i = 0; i < files.size(); ++i)
    {
        images.push_back(imread(files[i]));
        //originalImages.push_back(imread(files[i]));
    }   
    //
    // load directory of test text docs
    //
    vector<string> testFiles1;
    int loadTestFiles1 = loadDirectory(testFiles1, argv[2], argv[2],
                                        TEST_BOX_FILE_TYPE);
    // check that files were loaded
    if (loadTestFiles1 == EXIT_FAILURE)
    {
        return EXIT_FAILURE;
    }
    // hold names and Rects for test data for retrieval later
    Hands testBoxes;
    vector<Hands> allTestBoxes;
    for (int i = 0; i < testFiles1.size(); i++)
    {
        testBoxes.box = readTestBoxes(testFiles1[i]);
        size_t place = testFiles1[i].find_last_of("/\\");
        testBoxes.name = testFiles1[i].substr(place + 1, 2);
        allTestBoxes.push_back(testBoxes);
    }
    
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
    // create bounding boxes for images
    Mat frame;
    Hands imageBoxes;
    vector<Hands> allBoxes;
    for (int i = 0; i < images.size(); i++)
    {
        // Load images.
        frame = images[i];
        // add hand boxes to images
        imageBoxes.box = draw_bouding_boxes(frame, net, class_list);
        Mat segmented_hands = images[i].clone();
        Mat full_mask = Mat::zeros(Size(images[i].cols, images[i].rows), CV_8UC1);;
        for (int i = 0; i < imageBoxes.box.size(); i++)
        {
            // Doing segmentation on each cropped hand
            Mat cropped_hand = frame(imageBoxes.box[i]);
            Mat cropped_hand_seg = segmentHand(cropped_hand);

            segmented_hands = drawSegHands(segmented_hands, imageBoxes.box, cropped_hand_seg, i);
        }
        imageBoxes.handImage = segmented_hands;
        size_t place = files[i].find_last_of("/\\");
        imageBoxes.name = files[i].substr(place + 1, 2);
        // add ground truth boxes to images
        allBoxes.push_back(imageBoxes);
        int index = 0;
        while (allTestBoxes[index].name != allBoxes[i].name)
        {
            index++;
        }
        addGroundTruthBoxes(frame, allTestBoxes[index].box);
        
        // uncomment to save images
//        saveImage("../output_images", "boxes_" + imageBoxes.name, frame, ".jpg");
        
        // show images
        imshow("Segmented Hands Output", segmented_hands);
        //imshow("Hand Detection Output", frame);
        waitKey(0);
    }
    //
    // Performace Measurement Intersection over Union for hand detection
    //
    printIntersectionOverUnion(allBoxes, allTestBoxes);
    
    
    destroyAllWindows();
    return 0;
}
