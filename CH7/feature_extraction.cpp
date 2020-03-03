#include<iostream>
#include<opencv2/core.hpp>
#include<opencv2/features2d.hpp>
#include<opencv2/highgui.hpp>

using namespace std;

#define img1Location "./img1.jpg"
#define img2Location "./img2.jpg"
#define savePath "./imgFeature.jpg"

int main()
{
    cv::Mat img1 = cv::imread(img1Location);
    cv::Mat img2 = cv::imread(img2Location);  //图像读取
    
    cv::imshow("img1",img1);  //读取图像显示
    cv::waitKey(0);
    cv::imshow("img2",img2);
    cv::waitKey(0);

    std::vector<cv::KeyPoint>    keypoints1,keypoints2;  //特征点存储容器
    cv::Mat description1,description2;  //描述子
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);   //ORB初始化

    orb->detect(img1,keypoints1);  //特征点检测
    orb->detect(img2,keypoints2);

    orb->compute(img1,keypoints1,description1);  //描述子计算
    orb->compute(img2,keypoints2,description2);

    cv::Mat outimg1;
    cv::drawKeypoints(img1, keypoints1, outimg1,cv::Scalar::all(-1),cv::DrawMatchesFlags::DEFAULT);  //标注特征点到图片上
    cv::imshow("ORB特征点检测",outimg1);  //显示标注后的图片
    cv::waitKey(0);
    cv::imwrite(savePath,outimg1);

    vector<cv::DMatch> matchs;  //匹配存储器
    cv::BFMatcher matcher(cv::NORM_HAMMING);    //使用hamming匹配
    matcher.match(description1,description2,matchs);    //描述子 进行匹配

    double min_dist = 1000,max_dist = 0;    //匹配点距离极值
    for(int i = 0; i<description1.rows; i++)
    {
        double dist = matchs[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist>max_dist) max_dist = dist;
    }

    cout<<"--Max dist:"<<max_dist<<endl;
    cout<<"--Min dist:"<<min_dist<<endl;

    vector<cv::DMatch> goodMatchs;          //特征点筛选
    for( int i = 0;i < description1.rows;i++)
    {
        if(matchs[i].distance <= max(2*min_dist,30.0))
        {
            goodMatchs.push_back(matchs[i]);
        }
    }

    cv::Mat imgMatch;
    cv::Mat imgGoodMatch;
    cv::drawMatches(img1,keypoints1,img2,keypoints2,matchs,imgMatch);   //把匹配的特征点对画上
    cv::drawMatches(img1,keypoints1,img2,keypoints2,goodMatchs,imgGoodMatch);
    cv::imshow("所有匹配点结果",imgMatch);
    cv::imshow("优化匹配",imgGoodMatch);
    cv::waitKey(0);

    return 0;

}