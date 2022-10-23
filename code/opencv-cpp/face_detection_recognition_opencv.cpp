//=========================================================================
// Project: Image and live video stream processing using OpenCV C++ API
//= ========================================================================
// File : demo_3_6.cpp
// Author : Mohsen Ghazel(mghazel2020)
// Date : Mar 15, 2021
//=========================================================================
// Specifications:
//========================================================================= 
// To demonstrate OpenCV cascade classifier for face dtecetion and tracking 
// from live video stream.
//========================================================================== 
// History
//========================================================================== 
// Date                      Changes                        Author
//= ======================================================================== 
// Mar 15, 2021              Initial definition             mghazel2020
//========================================================================== 
// MIT License :
//========================================================================== 
// Copyright(c) 2021 mghazel2020
// References:
//========================================================================== 
//  https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
//  https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html
//========================================================================== 

#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/gapi/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"


#include <iostream>
#include <string>
#include <stdlib.h>
#include <io.h>
#include < direct.h >
#include <math.h>

using namespace cv;
using namespace std;

//============================================================================================== 
// output folder name
//==============================================================================================
// Output folder:
//==============================================================================================
// output folder name
string output_folder = "..\\results\\";
void detectAndDisplay(Mat frame);
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

int main(int argc, const char** argv)
{
    //-------------------------------------------------------------------------------------------
    // Create the output directory:
    //------------------------------------------------------------------------------------------
    // the output directory
    output_folder = output_folder + "Face_detection" + "\\";
    // create the output directory
    if (_mkdir(output_folder.c_str()) != 0)
    {
        cout << "Unable to create the output folder: " << output_folder << endl;
    }
    else
    {
        cout << "The output folder created successfully!" << output_folder << endl;
    }

    CommandLineParser parser(argc, argv,
        "{help h||}"
        "{face_cascade|data/haarcascades/haarcascade_frontalface_alt.xml|Path to face cascade.}"
        "{eyes_cascade|data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|Path to eyes cascade.}"
        "{camera|0|Camera device number.}");
    parser.about("\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n"
        "You can use Haar or LBP features.\n\n");
    parser.printMessage();
    String face_cascade_name = samples::findFile(parser.get<String>("face_cascade"));
    String eyes_cascade_name = samples::findFile(parser.get<String>("eyes_cascade"));
    //-- 1. Load the cascades
    if (!face_cascade.load(face_cascade_name))
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    };
    if (!eyes_cascade.load(eyes_cascade_name))
    {
        cout << "--(!)Error loading eyes cascade\n";
        return -1;
    };
    int camera_device = parser.get<int>("camera");
    VideoCapture capture;
    //-- 2. Read the video stream
    capture.open(camera_device);
    if (!capture.isOpened())
    {
        cout << "--(!)Error opening video capture\n";
        return -1;
    }
    Mat frame;
    while (capture.read(frame))
    {
        if (frame.empty())
        {
            cout << "--(!) No captured frame -- Break!\n";
            break;
        }
        //-- 3. Apply the classifier to the frame
        detectAndDisplay(frame);
        if (waitKey(10) == 27)
        {
            break; // escape
        }
    }
    return 0;
}
void detectAndDisplay(Mat frame)
{
    Mat frame_gray;
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    //-- Detect faces
    std::vector<Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces);
    //=================================================
    //-- Process the detected faces:
    //=================================================
    for (size_t i = 0; i < faces.size(); i++)
    {
        //---------------------------------------------------------------------
        // draw circles arond the detected faces
        //---------------------------------------------------------------------
        Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
        ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);
        //---------------------------------------------------------------------
        // draw bounding-boxes around the detected faces
        //---------------------------------------------------------------------
        // TLC
        Point tlc(faces[i].x, faces[i].y);
        // BRC
        Point brc(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
        // draw the bounding-box around the detected face
        rectangle(frame, tlc, brc, Scalar(0, 255, 255), 2, LINE_8);

        //=================================================
        //-- In each face, detect eyes
        //=================================================
        Mat faceROI = frame_gray(faces[i]);
        //-- In each face, detect eyes
        std::vector<Rect> eyes;
        eyes_cascade.detectMultiScale(faceROI, eyes);
        for (size_t j = 0; j < eyes.size(); j++)
        {
            //---------------------------------------------------------------------
            // draw circles arond the detected faces
            //---------------------------------------------------------------------
            Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
            int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
            circle(frame, eye_center, radius, Scalar(255, 0, 0), 4);
            //---------------------------------------------------------------------
            // draw bonding-boxes arond the detected faces
            //---------------------------------------------------------------------
            // TLC
            Point tlc(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y);
            // BRC
            Point brc(faces[i].x + eyes[j].x + eyes[j].width, faces[i].y + eyes[j].y + eyes[j].height);
            // draw the bounding-box around the detected face
            rectangle(frame, tlc, brc, Scalar(0, 255, 255), 2, LINE_8);
        }
    }
    //-- Show what you got
    imshow("Capture - Face detection", frame);
    //----------------------------------------------------
    // save the display
    //----------------------------------------------------
    imwrite(output_folder + "frame-face-detectio.jpg", frame);
    //----------------------------------------------------
    // Wait for a keystroke in the window
    waitKey(1000);
    // close all windows
    cv::destroyAllWindows();
    //----------------------------------------------------

}