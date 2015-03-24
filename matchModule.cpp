#include "featureModule.h"
#include "matchModule.h"

#include <Base/numeric/numericalalgorithm.h>
#include <Base/numeric/mat.h>
#include <Base/numeric/ransac.h>
#include <Base/numeric/vec.h>
#include <Base/Common/stopwatch.h>

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

#ifdef WIN32
#define PATH_SLASH '\\'
#define PATH_SLASH_X '/'
#define PATH_PARENT "..\\"
#else
#define PATH_SLASH '/'
#define PATH_SLASH_X '\\'
#define PATH_PARENT "../"
#endif

namespace tw
{
///
/// \brief GetMatchPathFromSiftFile: compose match file path (.mat) from sift file (.sift)
/// \param sift_path: the sift file path
/// \param is_output_path: whether we specify the --output_path parameter
/// \param output_path: the specified output directory
/// \return
///
std::string GetMatchPathFromSiftFile(const std::string & sift_path, std::string const & output_path = "")
{
    std::string filename_without_suffix = IOUtilities::SplitPathExt(sift_path).first;
    std::string sift_match_filename;

    if(output_path != "")      // output to the assigned folder
    {
        filename_without_suffix = IOUtilities::SplitPath(filename_without_suffix).second;
        sift_match_filename = IOUtilities::JoinPath(output_path, filename_without_suffix) + ".mat";
    }
    else        // output to the same folder as sift file
    {
        sift_match_filename = filename_without_suffix + ".mat";
    }
    return sift_match_filename;
}

bool WriteMatFile(int match_buf[][2], const int nmatch, std::string filename1, std::string filename2)
{
    ofstream fout;
    std::string dir = IOUtilities::SplitPath(filename1).first;
    std::string match_filename = dir + PATH_SLASH + "match_file.txt";
    fout.open(match_filename.c_str(), std::ios::app);
    if(!fout.is_open())
    {
        return false;
    }

    filename1 = IOUtilities::SplitPath(filename1).second;
    filename1 = IOUtilities::SplitPathExt(filename1).first + ".jpg";
    filename2 = IOUtilities::SplitPath(filename2).second;
    filename2 = IOUtilities::SplitPathExt(filename2).first + ".jpg";
    fout << filename1 << " " << filename2 << " " << nmatch << endl;
    for(int i = 0; i < nmatch; i++)
    {
        fout << match_buf[i][0] <<  " ";
    }
    fout << endl;
    for(int i = 0; i < nmatch; i++)
    {
        fout << match_buf[i][1] << " " ;
    }
    fout << endl;
    fout.close();
    return true;
}

///
/// \brief PairwiseMatching:do a pairwise matching between two image(sift) files, and save the match info in the .mat file.
///                         This includes a putative match function provided by SiftGPU and a guided match (using fundamental
///                         matrix and homography)
/// \param match_filename: the name of the output match file. This is mainly used by transction-style saving
/// \param first_sift: the sift data of the first sift file
/// \param second_sift: the sift data of the second sift file
/// \return
///
bool PairwiseMatching(const std::string &match_filename,
                      const std::string &first_sift_filename, const std::string &second_sift_filename,
                      tw::SiftData &first_sift, tw::SiftData &second_sift,
                      SiftMatchGPU &matcher, MatchParam const & match_param,
                      FILE * match_fd, zeta::TransactionLog<FILE *> & tlog)
{
    // write to the .mat file points correspondence and inlier information
    if(tlog.isReady())
    {
        {
            std::string transaction_name = "match_info" + first_sift_filename + "&" + second_sift_filename;
            zeta::Transaction<FILE *> transaction_filename(tlog, transaction_name);
            if(!transaction_filename.exist())
            {
                const int max_sift = 32768;
                int match_buf[max_sift][2];

                // set feature location and descriptors, this is needed by SiftGPU
                matcher.SetDescriptors(0, first_sift.getFeatureNum(), first_sift.getDesPointer());
                //matcher.SetFeatureLocation(0, first_sift.getLocPointer(), 3);
                matcher.SetDescriptors(1, second_sift.getFeatureNum(), second_sift.getDesPointer());
                //matcher.SetFeatureLocation(1, second_sift.getLocPointer(), 3);

                // get putative match and store the match result in match_buf
                int nmatch = matcher.GetSiftMatch(max_sift, match_buf);
                zeta::notify(zeta::Notify::Debug) << nmatch << " matches";
                if(nmatch < match_param.min_num_inlier) return true;

                // ransac fundamental matrix
                tw::FundamentalMatrixModel fundamental_model(match_buf, nmatch,
                                                             first_sift.getFeatureNum(), second_sift.getFeatureNum(),
                                                             first_sift.getLocPointer(), second_sift.getLocPointer(),
                                                             first_sift.getLocDim(), match_param.f_error0, match_param.f_error1);
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
                                                     first_sift.getFeatureNum(), second_sift.getFeatureNum(),
                                                     first_sift.getLocPointer(), second_sift.getLocPointer(),
                                                     first_sift.getLocDim(), match_param.h_error1);
                zeta::RANSACNaiveSampler homography_sampler(match_param.h_ransac_conf, match_param.h_ransac_ratio, 4, nmatch, 0);
                typedef zeta::RANSAC<tw::HomographyModel, zeta::RANSACNaiveSampler> RANSACHomography;
                RANSACHomography homography_ransac;
                RANSACHomography::Param homography_param(match_param.thread_num);
                std::vector<bool> homography_inlier_flags;
                int homography_inlier_num = homography_ransac(homography_sampler, homography_model, homography_inlier_flags, homography_param);
                double h_inlier_ratio = (double)homography_inlier_num / (double)nmatch;
                zeta::notify(zeta::Notify::Debug) << ", H[" << homography_inlier_num << "/" << nmatch << "]";

                if((f_inlier_ratio < match_param.min_f_inlier_ratio || fundamental_inlier_num < match_param.min_f_inlier)
                        && (h_inlier_ratio < match_param.min_h_inlier_ratio || homography_inlier_num < match_param.min_h_inlier))
                    return true;

                tw::SiftMatchPair sift_match(match_buf, nmatch,
                                             homography_inlier_flags, fundamental_inlier_flags,
                                             first_sift_filename, second_sift_filename,
                                             homography_model.getHomography(), fundamental_model.getFundmantalMatrix(),
                                             homography_inlier_num, fundamental_inlier_num);

                sift_match.WriteSiftMatchPair(match_fd);
            }
            else
            {
                zeta::notify(zeta::Notify::Debug) << "existed and skip";
                return false;
            }
        }
    }
    else
    {
        zeta::notify(zeta::Notify::Error) << "Transaction log is not ready\n";
    }

    return true;
}

}   // end of namespace tw;

bool InitializeMatching(std::string const & sift_list_path,
                        std::vector<std::string> & sift_filenames,
                        std::vector<tw::SiftData> & sift_data,
                        std::vector<size_t> & image_lengths,
                        std::string const & image_list_path = "")
{
    // Read image list
    std::vector<std::string> image_filenames;
    if(image_list_path != "")
    {
        if(!IOUtilities::FileExist(image_list_path))
        {
            zeta::notify(zeta::Notify::Warning) << "[SiftGPU Matching] Image list file does not exist: "
                                              << image_list_path << '\n';
        }
        else {
            zeta::notify(zeta::Notify::Gossip) << "[SiftGPU Matching] Read image list file: "
                                               << image_list_path << '\n';
            IOUtilities::ExtractNonEmptyLines(image_list_path, image_filenames);
        }
    }

    if(!IOUtilities::FileExist(sift_list_path))
    {
        zeta::notify(zeta::Notify::Error) << "[SiftGPU Matching] Sift list file does not exist: "
                                          << sift_list_path << '\n';
        return false;
    }
    else {
        IOUtilities::ExtractNonEmptyLines(sift_list_path, sift_filenames);
    }

    bool read_image_header = false;
    if(image_filenames.size() != sift_filenames.size())
    {
        if(image_list_path != "")
        {
            zeta::notify(zeta::Notify::Error) << "[SiftGPU Matching] The number of sift files and images are inconsistent.\n"
                                              << "# image files: " << image_filenames.size() << '\n'
                                              << "# sift files: " << sift_filenames.size() << '\n';
            return false;
        }
    }
    else {
        read_image_header = true;
    }

    int sift_num = sift_filenames.size();
    if(read_image_header)
    {
        for(size_t pid = 0; pid < sift_num; pid++)
        {
            std::string image_path = image_filenames[pid];
            if (IOUtilities::FileExist(image_path))
            {
                zeta::ImageHeader header;
                if (!zeta::loadImageHeader(header, image_path))
                {
                    zeta::notify(zeta::Notify::Warning) << "[SiftGPU Matching] Fail to load image: "
                                                        << image_path << '\n';
                    read_image_header = false;
                    break;
                }
                else {
                    zeta::notify(zeta::Notify::Debug) << "[SiftGPU Matching] Read image file: "
                                                      << image_path << '\n' << "Image size: "
                                                      << header.width() << " x " << header.height() << '\n';
                }

                size_t image_height = header.height();
                size_t image_width = header.width();
                if(image_width == 0 || image_height == 0)
                {
                    zeta::notify(zeta::Notify::Warning) << "[SiftGPU Matching] Fail to read image size: "
                                                        << image_path << '\n';
                    read_image_header = false;
                    break;
                }

                size_t image_length = std::max(image_width, image_height);
                image_lengths.push_back(image_length);
                cout << pid << " " << image_length << endl;
            }
        }
        zeta::notify(zeta::Notify::Gossip) << "[SiftGPU Matching] Read " << sift_num << " image headers.\n";
    }

    sift_data.resize(sift_num);
    for(size_t i = 0; i < sift_num; i++)
    {
        if(!sift_data[i].ReadSiftFile(sift_filenames[i]))
        {
            zeta::notify(zeta::Notify::Error) << "[Glsl Matching] Error: Fail to read the sift file: "
                                              << sift_filenames[i] << '\n';
            return false;
        }
        else {
            zeta::notify(zeta::Notify::Debug) << "[SiftGPU Matching] Read sift file: "
                                              << sift_filenames[i] << '\n' << "Sift count: "
                                              << sift_data[i].getFeatureNum() << '\n';
        }
    }
    zeta::notify(zeta::Notify::Gossip) << "[SiftGPU Matching] Read " << sift_num << " sift files.\n";

    return true;
}

bool FullMatch(std::vector<std::string> const & sift_filenames,
               std::vector<tw::SiftData> & sift_data,
               std::vector<size_t> const & image_lengths,
               std::string const & output_path,
               zeta::StopWatches & timers,
               MatchGlobalParam const & match_global_param)
{
    zeta::StopWatch match_time;
    match_time.Start();

    SiftMatchGPU matcher(match_global_param.max_sift);    //32768 is the maximum number of features to match
    {
        zeta::LocalWatch timer("Configure", timers);
        // choose language for matching
        if(match_global_param.cuda_device != -1 && match_global_param.cuda_device >= 0)   // use CUDA for matching
        {
            matcher.SetLanguage(3 + match_global_param.cuda_device);
        }
        else    // use default GLSL
        {
            //disabling this line for cuda matching
            if(matcher.CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED) return false;
        }
        if(matcher.VerifyContextGL() == 0)  return false;
    }

    size_t sift_num = sift_data.size();
    MatchParam match_param;

    for(size_t i = 0; i < sift_num; i++)
    {
        std::string sift_match_filename = tw::GetMatchPathFromSiftFile(sift_filenames[i], output_path);
        FILE * match_fd = NULL;
        zeta::TransactionLog<FILE *> tlog(match_fd, sift_match_filename);

        size_t valid_matches = 0;
        for(size_t j = i+1; j < sift_num; j++)
        {
            zeta::LocalWatch timer("Match", timers);
            long default_fill = std::cout.fill();
            zeta::notify(zeta::Notify::Debug) << std::setw(8) << std::setfill('0') << i << " and "
                                              << std::setw(8) << std::setfill('0') << j << ":\t";
            std::cout.fill(default_fill);

            if(image_lengths.size() == sift_num)
            {
                match_param.f_error0 = std::max(2.0, (double)image_lengths[i] * match_global_param.norm_f_error);
                match_param.f_error1 = std::max(2.0, (double)image_lengths[j] * match_global_param.norm_f_error);
                match_param.h_error0 = std::max(2.0, (double)image_lengths[i] * match_global_param.norm_f_error);
                match_param.h_error1 = std::max(2.0, (double)image_lengths[j] * match_global_param.norm_f_error);
            }

            if(tw::PairwiseMatching(sift_match_filename,
                                    sift_filenames[i], sift_filenames[j],
                                    sift_data[i], sift_data[j], matcher,
                                    match_param, match_fd, tlog))
            {
                valid_matches++;
            }

            zeta::notify(zeta::Notify::Debug) << ", " << zeta::StopWatch::ToSeconds(timer.elapsed()) << " sec\n";
        }

        zeta::notify(zeta::Notify::Gossip) << "[SiftGPU Matching] Detect matches: "
                                           << sift_filenames[i] << "\n[" << i + 1 << "/" << sift_num << "]: "
                                           << valid_matches << " matches computed.\n";
    }

    return true;
}

bool OptionalMatch(std::string const & optional_match_path
                   , std::vector<std::string> const & sift_filenames
                   , std::vector<tw::SiftData> & sift_data
                   , std::vector<size_t> const & image_lengths
                   , std::string const & output_path
                   , zeta::StopWatches & timers
                   , MatchGlobalParam const & match_global_param)
{
    SiftMatchGPU matcher(match_global_param.max_sift);    //32768 is the maximum number of features to match
    {
        zeta::LocalWatch timer("Configure", timers);
        // choose language for matching
        if(match_global_param.cuda_device != -1 && match_global_param.cuda_device >= 0)   // use CUDA for matching
        {
            matcher.SetLanguage(3 + match_global_param.cuda_device);
        }
        else    // use default GLSL
        {
            //disabling this line for cuda matching
            if(matcher.CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED) return false;
        }
        if(matcher.VerifyContextGL() == 0)  return false;
    }

    // Construct the sift lookup map
    size_t sift_num = sift_filenames.size();
    std::tr1::unordered_map<std::string, size_t> sift_file_map;
    for(size_t fid = 0; fid < sift_filenames.size(); fid++)
    {
        std::string sift_file = sift_filenames[fid];
        IOUtilities::Trim(sift_file);
        sift_file_map.insert(std::make_pair(sift_file, fid));
    }

    zeta::notify(zeta::Notify::Gossip) << "[SiftGPU Match] Read match pair file: "
                                       << optional_match_path << '\n';

    std::ifstream match_pair_stream(optional_match_path.c_str(), std::ios::in);
    if(!match_pair_stream.is_open())
    {
        zeta::notify(zeta::Notify::Error) << "[SiftGPU Match] Fail to read optional match file: "
                                          << optional_match_path << '\n';
        return false;
    }

    std::tr1::unordered_map<size_t, std::tr1::unordered_set<size_t> > match_pairs;
    size_t num_match_pairs = 0;
    std::string match_line;
    while(!match_pair_stream.eof())
    {
        std::getline(match_pair_stream, match_line);
        if(match_line.size() < 3)
            continue;

        std::stringstream match_string_stream;
        std::string first_file, second_file;
        size_t first_index, second_index;
        first_file = second_file = "";

        match_string_stream << match_line;
        match_string_stream >> first_file >> second_file;
        if(first_file == "" || second_file == "")
        {
            zeta::notify(zeta::Notify::Warning) << "[SiftGPU Match] Invalid sift pairs for matching: "
                                                << match_line << '\n';
            continue;
        }

        IOUtilities::Trim(first_file);
        if(!IOUtilities::FileExist(first_file))
        {
            zeta::notify(zeta::Notify::Warning) << "[SiftGPU Match] The sift file for matching does not exist: "
                                                << first_file << '\n';
            continue;
        }
        else {
            std::tr1::unordered_map<std::string, size_t>::const_iterator fitr = sift_file_map.find(first_file);
            if(fitr == sift_file_map.end())
            {
                zeta::notify(zeta::Notify::Warning) << "[SiftGPU Match] Illegal sift file for matching: "
                                                    << first_file << '\n';
                continue;
            }
            else {
                first_index = fitr->second;
            }
        }

        IOUtilities::Trim(second_file);
        if(!IOUtilities::FileExist(second_file))
        {
            zeta::notify(zeta::Notify::Warning) << "[SiftGPU Match] The sift file for matching does not exist: "
                                                << second_file << '\n';
            continue;
        }
        else {
            std::tr1::unordered_map<std::string, size_t>::const_iterator fitr = sift_file_map.find(second_file);
            if(fitr == sift_file_map.end())
            {
                zeta::notify(zeta::Notify::Warning) << "[SiftGPU Match] Illegal sift file for matching: "
                                                    << second_file << '\n';
                continue;
            }
            else {
                second_index = fitr->second;
            }
        }

        if(first_index >= second_index)
            continue;
        else
            num_match_pairs++;

        std::tr1::unordered_map<size_t, std::tr1::unordered_set<size_t> >::iterator itr = match_pairs.find(first_index);
        if(itr == match_pairs.end())
        {
            std::tr1::unordered_set<size_t> second_indexes;
            second_indexes.insert(second_index);
            match_pairs.insert(std::make_pair(first_index, second_indexes));
        }
        else {
            std::tr1::unordered_set<size_t> & second_indexes = itr->second;
            second_indexes.insert(second_index);
        }
    }
    zeta::notify(zeta::Notify::Debug) << "[SiftGPU Match] " << num_match_pairs << " valid match pairs.\n";

    MatchParam match_param;
    for(std::tr1::unordered_map<size_t, std::tr1::unordered_set<size_t> >::const_iterator itr0 = match_pairs.begin()
        ; itr0 != match_pairs.end(); itr0++)
    {
        size_t first_index = itr0->first;
        FILE * match_fd = NULL;
        std::string sift_match_filename = tw::GetMatchPathFromSiftFile(sift_filenames[first_index], output_path);
        zeta::TransactionLog<FILE *> tlog(match_fd, sift_match_filename);

        std::tr1::unordered_set<size_t> const & second_indexes = itr0->second;
        size_t valid_matches = 0;

        for(std::tr1::unordered_set<size_t>::const_iterator itr1 = second_indexes.begin()
            ; itr1 != second_indexes.end(); itr1++)
        {
            zeta::LocalWatch timer("Match", timers);
            size_t second_index = *itr1;
            long default_fill = std::cout.fill();
            zeta::notify(zeta::Notify::Debug) << std::setw(8) << std::setfill('0') << first_index << " and "
                                              << std::setw(8) << std::setfill('0') << second_index << ":\t";
            std::cout.fill(default_fill);

            if(image_lengths.size() == sift_num)
            {
                match_param.f_error0 = std::max(2.0, (double)image_lengths[first_index] * match_global_param.norm_f_error);
                match_param.f_error1 = std::max(2.0, (double)image_lengths[second_index] * match_global_param.norm_f_error);
                match_param.h_error0 = std::max(2.0, (double)image_lengths[first_index] * match_global_param.norm_f_error);
                match_param.h_error1 = std::max(2.0, (double)image_lengths[second_index] * match_global_param.norm_f_error);
            }

            std::string sift_match_filename = tw::GetMatchPathFromSiftFile(sift_filenames[first_index], output_path);
            if(!tw::PairwiseMatching(sift_match_filename,
                                     sift_filenames[first_index], sift_filenames[second_index],
                                     sift_data[first_index], sift_data[second_index],
                                     matcher, match_param, match_fd, tlog))
            {
                continue;
            }
            else {
                valid_matches++;
            }
            zeta::notify(zeta::Notify::Debug) << ", " << zeta::StopWatch::ToSeconds(timer.elapsed()) << " sec\n";
        }

        zeta::notify(zeta::Notify::Gossip) << "[SiftGPU Matching] Detect matches: " << sift_filenames[first_index]
                                           << "\n[" << first_index + 1 << "/" << sift_num << "]: "
                                           << valid_matches << " matches computed.\n";
    }

    return true;
}

bool GlslMatchHandler(std::vector<std::string> & commandOptions)
{
    zeta::StopWatches timers;
    zeta::OptionParser parser;
    if(!parser.registerOption("input_path", "", "input path")
            || !parser.registerOption("--output_path", "", "optional output path for .mat and .tlog files")
            || !parser.registerOption("--optional_match", "", "the path to a file that specifies the match pairs (.sift file)")
            || !parser.registerOption("--image_list", "", "the image list)")
            || !parser.registerOption("--cuda", "-1", "cuda device num 0 (for one GPU just use 0)")
            || !parser.registerOption("--thread_num", "0", "The number of threads for matching"))
    {
        return false;
    }

    if (!parser.parse(commandOptions))
    {
        return false;
    }

    MatchGlobalParam match_global_param;
    std::string sift_list_path = parser.getString("input_path");
    std::string output_path = parser.getString("--output_path");
    std::string optional_match_path = parser.getString("--optional_match");
    std::string image_list = parser.getString("--image_list");
    match_global_param.cuda_device = parser.getValue<int>("--cuda");
    match_global_param.thread_num = parser.getValue<int>("--thread_num");

    if(match_global_param.thread_num == 0)
    {
        match_global_param.thread_num = zeta::GetPhysicalCPUCount();
    }

    // handle optional parameters
    bool is_optional_match = false;
    if (optional_match_path != "")
    {
        if(IOUtilities::FileExist(optional_match_path))
        {
            is_optional_match = true;
        }
        else {
            zeta::notify(zeta::Notify::Warning) << "[SiftGPU Match] Fail to read match pair file: "
                                                << optional_match_path << '\n'
                                                << "[SiftGPU Match] Switch to full match.\n";
        }
    }

    // Initialize matching file
    std::vector<std::string> sift_filenames;
    std::vector<size_t> image_lengths;
    std::vector<tw::SiftData> sift_data;
    {
        zeta::LocalWatch timer("Read Sift", timers);
        if(!InitializeMatching(sift_list_path, sift_filenames, sift_data, image_lengths, image_list))
            return false;
    }

    if(!is_optional_match)  // full match O(n^2)
    {
        if(!FullMatch(sift_filenames, sift_data, image_lengths, output_path, timers, match_global_param))
            return false;
    }
    else    // partial match, look up sift_lisf for strings in optional_match_path
    {
        if(!OptionalMatch(optional_match_path, sift_filenames, sift_data, image_lengths, output_path, timers, match_global_param))
            return false;
    }

    zeta::notify(zeta::Notify::Gossip) << timers << '\n';
    return true;
}
