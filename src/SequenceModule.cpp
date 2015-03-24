#include "SequenceModule.h"

#include <Base/numeric/numericalalgorithm.h>
#include <Base/numeric/mat.h>
#include <Base/numeric/ransac.h>
#include <Base/numeric/vec.h>
#include <Base/Common/stopwatch.h>
#include <Base/Common/cmdframework.h>
#include <Base/Common/ioutilities.h>

#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <time.h>
#include <cmath>
#include <numeric>

#define DTYPE unsigned char
#define LTYPE float

using std::cout;
using std::endl;

bool LoadSiftData(std::vector<std::string> &sift_filenames,
                  std::vector<std::string> &image_filenames,
                  std::vector<tw::SiftData> &sift_data,
                  std::vector<size_t> &image_lengths)
{
    int sift_num = sift_filenames.size();
    if(sift_filenames.size() != image_filenames.size())
    {
        zeta::notify(zeta::Notify::Error) << "[LoadSiftData]#image and #sift are not consistent.\n";
        return false;
    }
    int read_image_count = 0;
    for(size_t pid = 0; pid < sift_num; pid++)
    {
        std::string image_path = image_filenames[pid];
        if (IOUtilities::FileExist(image_path))
        {
            zeta::ImageHeader header;
            if (!zeta::loadImageHeader(header, image_path))
            {
                zeta::notify(zeta::Notify::Warning) << "[LoadSiftData]Fail to load image: "
                                                    << image_path << '\n';
                break;
            }
            else
            {
                zeta::notify(zeta::Notify::Debug) << "[LoadSiftData] Read image file: "
                                                  << image_path << '\n' << "Image size: "
                                                  << header.width() << " x " << header.height() << '\n';
            }

            size_t image_height = header.height();
            size_t image_width = header.width();
            if(image_width == 0 || image_height == 0)
            {
                zeta::notify(zeta::Notify::Warning) << "[LoadSiftData] Fail to read image size: "
                                                    << image_path << '\n';
                break;
            }

            size_t image_length = std::max(image_width, image_height);
            image_lengths.push_back(image_length);
            read_image_count++;
        }
    }
    zeta::notify(zeta::Notify::Gossip) << "[LoadSiftData] Read " << read_image_count << " image headers.\n";

    sift_data.resize(sift_num);
    size_t MAX_SIFT_NUM = 0;
    for(size_t i = 0; i < sift_num; i++)
    {
        if(!sift_data[i].ReadSiftFile(sift_filenames[i]))
        {
            zeta::notify(zeta::Notify::Error) << "[LoadSiftData] Error: Fail to read the sift file: "
                                              << sift_filenames[i] << '\n';
            return false;
        }
        else
        {
            zeta::notify(zeta::Notify::Debug) << "[LoadSiftData] Read sift file: "
                                              << sift_filenames[i] << '\n' << "Sift count: "
                                              << sift_data[i].getFeatureNum() << '\n';
            if(MAX_SIFT_NUM < sift_data[i].getFeatureNum())
            {
                MAX_SIFT_NUM = sift_data[i].getFeatureNum();
            }
        }
    }
    zeta::notify(zeta::Notify::Gossip) << "[LoadSiftData] Read " << sift_num << " sift files.\n"
                                       << "Maximum sift num is " << MAX_SIFT_NUM << "\n";
    if(sift_num != read_image_count)
    {
        zeta::notify(zeta::Notify::Warning) << "[LoadSiftData]" << "#sift_file is larger than #loaded_image.\n";
    }

    return true;
}

bool DatasetAnalysis(std::vector<std::string> &sift_filenames,
                     std::vector<tw::SiftData> &sift_data,
                     std::vector<size_t> &image_lengths,
                     std::vector<tw::LinkNode> * &adj_lists,
                     const int window_size = 3)
{
    // initilize analysis
    const int image_num = sift_filenames.size();
    MatchGlobalParam match_global_param;
    MatchParam match_param;
    if(adj_lists != NULL) delete [] adj_lists;
    adj_lists = new std::vector<tw::LinkNode> [image_num];

    // initialize SiftGPU Matcher
    const int max_sift = 32768;
    SiftMatchGPU matcher(max_sift);
    if(matcher.CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED) return false;
    if(matcher.VerifyContextGL() == 0)  return false;

    // sequence matching
    const int MAX_WINDOW_SIZE = window_size;
    int computed_pair_num = 0;
    int matched_pair_num = 0;
    for(int i = 0; i < image_num; i++)
    {
        for(int j = i+1, wc = 0; j < image_num && wc < MAX_WINDOW_SIZE; j++, wc++)
        {
            computed_pair_num++;
            int match_buf[max_sift][2];

            cout << IOUtilities::SplitPath(sift_filenames[i]).second << " " << IOUtilities::SplitPath(sift_filenames[j]).second << "\n";
            matcher.SetDescriptors(0, sift_data[i].getFeatureNum(), sift_data[i].getDesPointer());
            matcher.SetDescriptors(1, sift_data[j].getFeatureNum(), sift_data[j].getDesPointer());

            // get putative match and store the match result in match_buf
            int nmatch = matcher.GetSiftMatch(max_sift, match_buf);
            zeta::notify(zeta::Notify::Debug) << nmatch << " matches\n";

            tw::LinkNode lnode(i, j, nmatch);
            if(nmatch > match_param.min_num_inlier)
            {
                // set threshold
                match_param.f_error0 = std::max(2.0, (double)image_lengths[i] * match_global_param.norm_f_error);
                match_param.f_error1 = std::max(2.0, (double)image_lengths[j] * match_global_param.norm_f_error);
                match_param.h_error0 = std::max(2.0, (double)image_lengths[i] * match_global_param.norm_f_error);
                match_param.h_error1 = std::max(2.0, (double)image_lengths[j] * match_global_param.norm_f_error);

                // ransac fundamental matrix
                tw::FundamentalMatrixModel fundamental_model(match_buf, nmatch,
                                                             sift_data[i].getFeatureNum(), sift_data[j].getFeatureNum(),
                                                             sift_data[i].getLocPointer(), sift_data[j].getLocPointer(),
                                                             sift_data[i].getLocDim(), match_param.f_error0, match_param.f_error1);
                zeta::RANSACNaiveSampler fundamental_sampler(match_param.f_ransac_conf, match_param.f_ransac_ratio, 8, nmatch, 0);
                typedef zeta::RANSAC<tw::FundamentalMatrixModel, zeta::RANSACNaiveSampler> RANSACFundamental;
                RANSACFundamental ransac_fundamental;
                RANSACFundamental::Param fundamental_param(match_param.thread_num);
                std::vector<bool> fundamental_inlier_flags;
                int fundamental_inlier_num = ransac_fundamental(fundamental_sampler, fundamental_model, fundamental_inlier_flags, fundamental_param);
                double f_inlier_ratio = (double)fundamental_inlier_num / (double)nmatch;
                zeta::notify(zeta::Notify::Debug) << ", F[" << fundamental_inlier_num << "/" << nmatch << "]";

                // ransac homography
                tw::HomographyModel homography_model(match_buf, nmatch,
                                                     sift_data[i].getFeatureNum(), sift_data[j].getFeatureNum(),
                                                     sift_data[i].getLocPointer(), sift_data[j].getLocPointer(),
                                                     sift_data[i].getLocDim(), match_param.h_error1);
                zeta::RANSACNaiveSampler homography_sampler(match_param.h_ransac_conf, match_param.h_ransac_ratio, 4, nmatch, 0);
                typedef zeta::RANSAC<tw::HomographyModel, zeta::RANSACNaiveSampler> RANSACHomography;
                RANSACHomography homography_ransac;
                RANSACHomography::Param homography_param(match_param.thread_num);
                std::vector<bool> homography_inlier_flags;
                int homography_inlier_num = homography_ransac(homography_sampler, homography_model, homography_inlier_flags, homography_param);
                double h_inlier_ratio = (double)homography_inlier_num / (double)nmatch;
                zeta::notify(zeta::Notify::Debug) << ", H[" << homography_inlier_num << "/" << nmatch << "]\n";

                // TODO(tianwei): find a better scoring function
                if(fundamental_inlier_num > match_param.min_f_inlier)
                {
                    lnode.score += 0.8;
                    lnode.v_match = fundamental_inlier_num;
                    matched_pair_num++;
                }
                if(homography_inlier_num > match_param.min_h_inlier)
                {
                    lnode.score += 0.8;
                    lnode.score = lnode.score > 1 ? 1 : lnode.score;
                }
                //save result in the adjacent lists
                adj_lists[i].push_back(lnode);
                tw::LinkNode lnode_inv = lnode;
                lnode_inv.dst = lnode.src; lnode_inv.src = lnode.dst;
                adj_lists[j].push_back(lnode_inv);
            }
        }
    }

    cout << "window size: " << window_size << endl;
    cout << "matched percent: " << (float) matched_pair_num/computed_pair_num
         << "[" << matched_pair_num << "/" << computed_pair_num << "]\n";
    return true;
}

///
/// \brief ConnectedComponent: compute the number of connected components
/// \param n: the number of nodes in the graph
/// \param match_result: adjacent lists
/// \return
///
int ConnectedComponent(const int n, std::vector<tw::LinkNode> *match_result)
{
    bool is_visited[n];
    memset(is_visited, false, sizeof(is_visited));
    int numCC = 0;
    for(int i = 0; i < n; i++)
    {
        if(!is_visited[i])
        {
            numCC++;
            std::queue<int> index_queue;
            is_visited[i] = true;
            index_queue.push(i);
            int component_size = 1;
            while(!index_queue.empty())
            {
                int curr = index_queue.front();
                index_queue.pop();
                for(int j = 0; j < match_result[curr].size(); j++)
                {
                    if(!is_visited[match_result[curr][j].dst])
                    {
                        is_visited[match_result[curr][j].dst] = true;
                        component_size++;
                        index_queue.push(match_result[curr][j].dst);
                    }
                }
            }
        }
    }
    return numCC;
}

bool SaveGraphResult(std::vector<tw::LinkNode> adj_lists)
{
    return true;
}

///
/// \brief SequenceMatchHandler: entry point for sequence module
/// \param commandOptions
/// \return
///
bool SequenceMatchHandler(std::vector<std::string> & commandOptions)
{
    zeta::OptionParser parser;
    if(!parser.registerOption("input_path", "", "input path")
            || !parser.registerOption("--output_path", "", "optional output path for .mat and .tlog files")
            || !parser.registerOption("--window_size", "3", "sequence match window size"))
    {
        return false;
    }

    if (!parser.parse(commandOptions))
    {
        zeta::notify(zeta::Notify::Error) << "Can't parse commandOptions\n";
        return false;
    }

    std::string image_list_path = parser.getString("input_path");
    std::string output_path = parser.getString("--output_path");
    int window_size = parser.getValue<int>("--window_size");

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

    std::vector<std::string> sift_filenames;
    for(int i = 0; i < image_filenames.size(); i++)
    {
        std::string sift_filename_tmp = IOUtilities::SplitPathExt(image_filenames[i]).first + ".sift";
        sift_filenames.push_back(sift_filename_tmp);
    }

    // Load data and image length info
    std::vector<size_t> image_lengths;
    std::vector<tw::SiftData> sift_data;
    if(!LoadSiftData(sift_filenames, image_filenames, sift_data, image_lengths))
    {
        zeta::notify(zeta::Notify::Error) << "Can't load sift data.\n";
        return false;
    }

    std::vector<tw::LinkNode> *adj_lists = NULL;
    tw::ImageDatasetInfo image_data(image_filenames, sift_filenames, adj_lists);

    if(!DatasetAnalysis(sift_filenames, sift_data, image_lengths, adj_lists, window_size))
    {
        zeta::notify(zeta::Notify::Error) << "Can't analysis dataset.\n";
        return false;
    }

    int numCC = ConnectedComponent(sift_filenames.size(), adj_lists);
    cout << "number of connected components: " << numCC << endl;

    delete [] adj_lists;
    return true;
}
