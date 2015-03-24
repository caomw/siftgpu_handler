#include "ShowModule.h"
#include "featureModule.h"
#include "matchModule.h"

#include <Base/numeric/numericalalgorithm.h>
#include <Base/numeric/mat.h>
#include <Base/numeric/ransac.h>
#include <Base/numeric/vec.h>
#include <Base/Common/stopwatch.h>
#include <Base/Common/cmdframework.h>
#include <Base/Common/ioutilities.h>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <time.h>
#include <cmath>
#include <numeric>

using std::cout;
using std::endl;

namespace tw
{

int GetImageLength(std::string image_path)
{
    zeta::ImageHeader header;
    if (!zeta::loadImageHeader(header, image_path))
    {
        zeta::notify(zeta::Notify::Error) << "[SiftGPU Matching] Fail to load image: "
                                            << image_path << '\n';
    }
    else
    {
        zeta::notify(zeta::Notify::Debug) << "[SiftGPU Matching] Read image file: "
                                          << image_path << '\n' << "Image size: "
                                          << header.width() << " x " << header.height() << '\n';
    }

    int image_height = header.height();
    int image_width = header.width();
    if(image_width == 0 || image_height == 0)
    {
        zeta::notify(zeta::Notify::Warning) << "[SiftGPU Matching] Fail to read image size: "
                                            << image_path << '\n';
    }

    int image_length = std::max(image_width, image_height);
    return image_length;
}

bool ConvertSift2Opencv(tw::SiftData &sift, std::vector<cv::KeyPoint> &keypoints)
{
    int feature_num = sift.getFeatureNum();
    int loc_dim = sift.getLocDim();     //x, y, color, scale, orientation

    LTYPE *ploc = sift.getLocPointer();
    keypoints.resize(feature_num);
    for(int i = 0; i < feature_num; i++)
    {
        int idx = i * loc_dim;
        cv::Point2f pt(ploc[idx], ploc[idx+1]);
        cv::KeyPoint kp(pt, ploc[idx+3], ploc[idx+4]);
        keypoints[i] = kp;
    }

    return true;
}

} // end of namespace tw

bool ShowMatchHandler(std::vector<std::string> & commandOptions)
{
    // parse argv
    zeta::OptionParser parser;
    if(!parser.registerOption("input_path", "", "input path")
            ||!parser.registerOption("--image_length", "0", "image length (used for setting threshold"))
    {
        return false;
    }

    if (!parser.parse(commandOptions))
    {
        zeta::notify(zeta::Notify::Error) << "Can't parse commandOptions\n";
        return false;
    }

    std::string image_list_path = parser.getString("input_path");
    int image_length_default = parser.getValue<int>("image_length");

    // Extract image filenames and sift filenames
    std::vector<std::string> image_filenames;
    if(!IOUtilities::FileExist(image_list_path))
    {
        zeta::notify(zeta::Notify::Error) << "Image list file does not exist: " << image_list_path << '\n';
        return false;
    }
    else
    {
        zeta::notify(zeta::Notify::Gossip) << "Read image list file: " << image_list_path << '\n';
        IOUtilities::ExtractNonEmptyLines(image_list_path, image_filenames);
    }

    if(image_filenames.size() < 2)
    {
        zeta::notify(zeta::Notify::Error) << "Image list should contain at least 2 images.\n";
        return false;
    }

    // read from the image list and extract the first two images for visualization
    std::string image_path1 = image_filenames[0];
    std::string image_path2 = image_filenames[1];

    std::string sift_filename1 = IOUtilities::SplitPathExt(image_filenames[0]).first + ".sift";
    std::string sift_filename2 = IOUtilities::SplitPathExt(image_filenames[1]).first + ".sift";

    // read sift data from sift files
    tw::SiftData sift_data1, sift_data2;
    if(!sift_data1.ReadSiftFile(sift_filename1) || !sift_data2.ReadSiftFile(sift_filename2))
    {
        zeta::notify(zeta::Notify::Error) << "[Glsl Matching] Error: Fail to read the sift file.";
    }

    // get image lengths(max(width, length) for setting the threshold)
    int image1_length = image_length_default;       //tw::GetImageLength(image_path1);
    int image2_length = image_length_default;       //tw::GetImageLength(image_path2);

    cv::Mat image1 = cv::imread(image_path1, CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat image2 = cv::imread(image_path2, CV_LOAD_IMAGE_GRAYSCALE);

    if(!image1.data || !image2.data)
    {
        zeta::notify(zeta::Notify::Error) << "Error reading image.\n";
        return false;
    }

    // convert keypoints into opencv format
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    tw::ConvertSift2Opencv(sift_data1, keypoints1);
    tw::ConvertSift2Opencv(sift_data2, keypoints2);

    // compute putative matches and guided matches
    MatchGlobalParam match_global_param;
    MatchParam match_param;
    SiftMatchGPU matcher(match_global_param.max_sift);
    if(matcher.CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED) return false;
    if(matcher.VerifyContextGL() == 0)  return false;

    // set match parameters
    if(image1_length != 0)
    {
        match_param.f_error0 = std::max(2.0, (double)image1_length * match_global_param.norm_f_error);
        match_param.f_error1 = std::max(2.0, (double)image2_length * match_global_param.norm_f_error);
        match_param.h_error0 = std::max(2.0, (double)image1_length * match_global_param.norm_f_error);
        match_param.h_error1 = std::max(2.0, (double)image2_length * match_global_param.norm_f_error);
    }


    int match_buf[match_global_param.max_sift][2];
    matcher.SetDescriptors(0, sift_data1.getFeatureNum(), sift_data1.getDesPointer());
    matcher.SetDescriptors(1, sift_data2.getFeatureNum(), sift_data2.getDesPointer());
    int nmatch = matcher.GetSiftMatch(match_global_param.max_sift, match_buf);
    zeta::notify(zeta::Notify::Gossip) << nmatch << " putative matches\n";

    // convert putative matches to opencv format
    std::vector<cv::DMatch> putative_matches;
    putative_matches.resize(nmatch);
    for(int i = 0; i < nmatch; i++)
    {
        cv::DMatch Dtemp(match_buf[i][0], match_buf[i][1], 0.0);
        putative_matches[i] = Dtemp;
    }
    cv::Mat img_putative_matches;
    cv::drawMatches(image1, keypoints1, image2, keypoints2,
                    putative_matches, img_putative_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::namedWindow("putative matches", cv::WINDOW_NORMAL);
    cv::imshow("putative matches", img_putative_matches);
    cv::imwrite("putative_match.jpg", img_putative_matches);
    cv::waitKey(0);

    // guided match (fundamental matrix verification)
    tw::FundamentalMatrixModel fundamental_model(match_buf, nmatch,
                                                 sift_data1.getFeatureNum(), sift_data2.getFeatureNum(),
                                                 sift_data1.getLocPointer(), sift_data2.getLocPointer(),
                                                 sift_data1.getLocDim(), match_param.f_error0, match_param.f_error1);
    zeta::RANSACNaiveSampler fundamental_sampler(match_param.f_ransac_conf, match_param.f_ransac_ratio, 8, nmatch, 0);
    typedef zeta::RANSAC<tw::FundamentalMatrixModel, zeta::RANSACNaiveSampler> RANSACFundamental;
    RANSACFundamental ransac_fundamental;
    RANSACFundamental::Param fundamental_param(match_param.thread_num);
    std::vector<bool> fundamental_inlier_flags;
    int fundamental_inlier_num = ransac_fundamental(fundamental_sampler, fundamental_model, fundamental_inlier_flags, fundamental_param);
    zeta::notify(zeta::Notify::Gossip) << fundamental_inlier_num << " guided matches (fundamental matrix)\n";

    std::vector<cv::DMatch> guided_matches;
    guided_matches.resize(fundamental_inlier_num);
    int counter = 0;
    for(int i = 0; i < nmatch; i++)
    {
        if(fundamental_inlier_flags[i])
        {
            cv::DMatch Dtemp(match_buf[i][0], match_buf[i][1], 0.0);
            guided_matches[counter++] = Dtemp;
        }
    }
    cout << counter << endl;
    cv::Mat img_guided_matches;
    cv::drawMatches(image1, keypoints1, image2, keypoints2,
                    guided_matches, img_guided_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                    std::vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::namedWindow("guided matches", cv::WINDOW_NORMAL);
    cv::imshow("guided matches", img_guided_matches);
    cv::imwrite("guided_match.jpg", img_guided_matches);
    cv::waitKey(0);

    return true;
}
