#include <stdio.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>

#include "retinaface.h"

int32_t NowMicros();

int main(int argc, char **argv)
{
    RetinaFace facedetect;
    string modelpath = "models/FaceDetector.mnn"; 
    facedetect.Initial(modelpath);

    cv::Mat bgr;
    string file = "test.jpg";
    bgr = cv::imread(file);
    vector<FaceObject> faces;
    int start = NowMicros();
    facedetect.DetectFace(bgr, faces);
    int end = NowMicros();
    std::cout << "time cost: " << (end - start) << "ms" << std::endl;

    for (size_t i = 0; i < faces.size(); i++)
    {
        FaceObject face = faces[i];
        cv::rectangle(bgr, cv::Point(face.rect.x, face.rect.y),
                      cv::Point(face.rect.x + face.rect.width, face.rect.y + face.rect.height),
                      cv::Scalar(0, 255, 0), 2);
        string score = to_string(face.prob);
        cv::putText(bgr, score, cv::Point(face.rect.x, face.rect.y), cv::FONT_HERSHEY_DUPLEX, 0.8, cv::Scalar(0, 255, 255));
    }
    cv::imshow("results", bgr);
    cv::waitKey(0);
}
