#include<iostream>
#include<chrono>
using namespace std;

#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/stitching.hpp>
#include<opencv2/imgproc.hpp>

#define moviePath "/home/studyhooligen/视频/home.3gp"
#define savePath "./saveVedio.3gp"

int main()
{
    cv::VideoCapture capVideo(moviePath);
    if(capVideo.isOpened())
    {
        cout<<"视频文件成功打开"<<endl;
    }
    else
    {
        cout<<"视频文件打开失败"<<endl;
        return 0;
    }

    cv::Size capVideoSize = cv::Size( (int)capVideo.get(cv::CAP_PROP_FRAME_WIDTH),  (int)capVideo.get(cv::CAP_PROP_FRAME_HEIGHT) );
    cout<< "hight of Video is:"<<capVideoSize.height
        <<",width is :"<<capVideoSize.width
        <<",totally frame Number:"<<capVideo.get(cv::CAP_PROP_FRAME_COUNT)
        <<",fps:"<<capVideo.get(cv::CAP_PROP_FPS)<<endl;

    cv::Size flipVideoSize = cv::Size(capVideoSize.width,capVideoSize.height*2);
    cv::VideoWriter saveVideoStream(savePath,capVideo.get(cv::CAP_PROP_FOURCC),capVideo.get(cv::CAP_PROP_FPS),flipVideoSize);
    
    cout<< "hight of Video is:"<<capVideo.get(cv::CAP_PROP_FRAME_HEIGHT)
        <<",width is :"<<capVideo.get(cv::CAP_PROP_FRAME_WIDTH)
        <<",totally frame Number:"<<capVideo.get(cv::CAP_PROP_FRAME_COUNT)
        <<",fps:"<<capVideo.get(cv::CAP_PROP_FPS)<<endl;

    cv::Mat singleFrame;
    cv::Mat flipFrame;
    cv::Mat combine;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    while (1)
    {
        capVideo>>singleFrame;
        if(singleFrame.empty()) break;
        cv::flip(singleFrame,flipFrame,1);
        cv::flip(flipFrame,flipFrame,0);

        // cv::imshow("srcFrame",singleFrame);
        // cv::imshow("flipFrame",flipFrame);
        // cv::waitKey(0);
        cv::vconcat(singleFrame,flipFrame,combine);
        //cv::imshow("combine",combine);
        //cv::waitKey(0);
        saveVideoStream<<combine;
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> (t2-t1);
    cout<<"处理用时:"<<time_used.count()<<"s."<<endl;

    saveVideoStream.release();
    
    return 0;
    
}