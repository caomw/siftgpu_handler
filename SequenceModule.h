#ifndef SEQUENCEMODULE_H
#define SEQUENCEMODULE_H

#include "matchModule.h"
#include "featureModule.h"
#include <Base/numeric/ransac.h>
#include <Base/numeric/vec.h>

#include <cstdlib>
#include <vector>
#include <string>

namespace tw
{

struct LinkNode
{
    int src;
    int dst;
    int i_match;
    int v_match;
    float score;

    LinkNode(int src_ = -1, int dst_ = -1, int i_match_ = 0, int v_match_ = 0, float score_ = 0.0):
        src(src_), dst(dst_), i_match(i_match_), v_match(v_match), score(score_) {}
};

class ImageDatasetInfo
{
public:
    ImageDatasetInfo()
    {
        size_ = 0;
    }

    ImageDatasetInfo(const std::vector<std::string> &image_filenames,
                     const std::vector<std::string> &sift_filenames,
                     std::vector<LinkNode> *adj_lists)
    {
        size_ = image_filenames.size();
        image_filenames_ = image_filenames;
        sift_filenames_ = sift_filenames;
        //adj_lists_ = adj_lists;
    }

private:
    size_t size_;
    std::vector<std::string> image_filenames_;
    std::vector<std::string> sift_filenames_;
    //std::vector<tw::SiftData> sift_data_; // TODO(tianwei): there is a double-free error, and a segmentation fault error.
    std::vector<LinkNode> *adj_lists_;
};

} // end of namespace

#endif // SEQUENCEMODULE_H
