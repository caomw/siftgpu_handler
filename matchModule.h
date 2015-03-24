#ifndef MATCHHANDLER_H
#define MATCHHANDLER_H

#include <Base/Common/notify.h>
#include <Base/geometry/bdlgeometry.h>
#include <Base/misc/imagehelper.h>
#include <Base/misc/phototourismhelper.h>
#include <Base/Common/cmdframework.h>
#include <Base/Common/ioutilities.h>
#include <Base/misc/formatconvertor.h>

#include <siftgpu/SiftGPU.h>
#include <siftgpu/FeaturePoints.h>
#include <siftgpu/MatchFile.h>
#include <siftgpu/points.h>

using std::cout;
using std::endl;

struct MatchParam
{
    MatchParam()
    {
        thread_num = 1;

        min_num_inlier = 25;
        min_f_inlier = 15;
        min_h_inlier = 15;

        min_f_inlier_ratio = 0.1;
        min_h_inlier_ratio = 0.1;
        f_ransac_ratio = 0.5;
        h_ransac_ratio = 0.5;
        f_ransac_conf = 0.999;
        h_ransac_conf = 0.999;

        f_error0 = 8;
        f_error1 = 8;
        h_error0 = 8;
        h_error1 = 8;
    }

    size_t thread_num;

    size_t min_num_inlier;
    size_t min_f_inlier;
    size_t min_h_inlier;

    double min_f_inlier_ratio;
    double min_h_inlier_ratio;
    double f_ransac_ratio;
    double h_ransac_ratio;
    double f_ransac_conf;
    double h_ransac_conf;

    double f_error0;
    double f_error1;
    double h_error0;
    double h_error1;
};

struct MatchGlobalParam
{
    MatchGlobalParam()
    {
        cuda_device = -1;
        thread_num = 0;
        max_sift = 32768;
        norm_f_error = 0.00285;
        norm_h_error = 0.00285;
    }

    int cuda_device;
    size_t thread_num;
    size_t max_sift;
    double norm_f_error;
    double norm_h_error;
};

struct FeatureMatchPair
{

public:
    enum FlagType { OUTLIER, HINLIER, FINLIER, INLIER};
    int first;
    int second;
    char flag;

    bool operator==(const FeatureMatchPair & rhs)
    {
        if(first == rhs.first && second == rhs.second && flag == rhs.flag) { return true; }
        else { return false; }
    }

    FeatureMatchPair & operator=(const FeatureMatchPair & rhs)
    {
        first = rhs.first;
        second = rhs.second;
        flag = rhs.flag;
        return *this;
    }

    void showInfo() const
    {
        zeta::notify(zeta::Notify::Gossip) << first << " " << second << " ";
        switch(flag)
        {
        case OUTLIER: zeta::notify(zeta::Notify::Gossip) << "OUTLIER\n";break;
        case HINLIER: zeta::notify(zeta::Notify::Gossip) << "HINLIER\n";break;
        case FINLIER: zeta::notify(zeta::Notify::Gossip) << "FINLIER\n";break;
        case INLIER: zeta::notify(zeta::Notify::Gossip) << "INLIER\n";break;
        default: zeta::notify(zeta::Notify::Error) << "ERROR\n";break;
        }
    }
};

namespace tw {
class SiftMatchPair
{
private:
    FeatureMatchPair *match_pairs_;
    std::string filename1_;
    std::string filename2_;
    int nmatch_;
    int homography_inlier_num_;
    int fundamental_inlier_num_;
    zeta::Mat3d homography_;
    zeta::Mat3d fundamental_matrix_;

public:
    SiftMatchPair(std::string filename1)
    {
        filename1_ = filename1;
        match_pairs_ = NULL;
    }

    SiftMatchPair(int match_buf[][2], const int nmatch
                    , const std::vector<bool>& homography_inlier_flag, const std::vector<bool>& fundamental_inlier_flag
                    , std::string filename1, std::string filename2
                    , const zeta::Mat3d &homography, const zeta::Mat3d &fundamental_matrix
                    , int homography_inlier_num, int fundamental_inlier_num)
    {
        match_pairs_ = new FeatureMatchPair[nmatch];
        nmatch_ = nmatch;
        filename1_ = filename1;
        filename2_ = filename2;
        homography_ = homography;
        fundamental_matrix_ = fundamental_matrix;
        homography_inlier_num_ = homography_inlier_num;
        fundamental_inlier_num_ = fundamental_inlier_num;

        for(size_t i = 0; i < nmatch_; i++)
        {
            match_pairs_[i].first = match_buf[i][0];
            match_pairs_[i].second = match_buf[i][1];
            match_pairs_[i].flag = (FeatureMatchPair::OUTLIER | homography_inlier_flag[i] | (fundamental_inlier_flag[i] << 1));
        }
    }

    // copy constructor
    SiftMatchPair(const SiftMatchPair &obj):
        filename1_(obj.filename1_), filename2_(obj.filename2_),
        nmatch_(obj.nmatch_), homography_inlier_num_(obj.homography_inlier_num_), fundamental_inlier_num_(obj.fundamental_inlier_num_),
        homography_(obj.homography_), fundamental_matrix_(obj.fundamental_matrix_)
    {
        FeatureMatchPair *match_pairs_origin = match_pairs_;
        match_pairs_ = new FeatureMatchPair [obj.nmatch_];
        for(int i = 0; i < obj.nmatch_; i++)
        {
            match_pairs_[i] = obj.match_pairs_[i];
        }
        delete [] match_pairs_origin;
    }

    bool operator==(const SiftMatchPair &rhs)
    {   //NOTICE: this is a incomplete comparison
        if(this->filename1_ == rhs.filename1_ && this->filename2_ == rhs.filename2_ &&
                this->nmatch_ == rhs.nmatch_ &&
                this->fundamental_inlier_num_ == rhs.fundamental_inlier_num_ &&
                this->homography_inlier_num_ == rhs.homography_inlier_num_) { return true; }
        else { return false; }
    }

    // copy assignment operator
    SiftMatchPair & operator=(const SiftMatchPair & rhs)
    {
        if(*this == rhs)return *this;
        filename1_ = rhs.filename1_;
        filename2_ = rhs.filename2_;
        nmatch_ = rhs.nmatch_;
        homography_inlier_num_ = rhs.homography_inlier_num_;
        fundamental_inlier_num_ = rhs.fundamental_inlier_num_;
        homography_ = rhs.homography_;
        fundamental_matrix_ = rhs.fundamental_matrix_;
        FeatureMatchPair *originMatchPairs = match_pairs_;
        match_pairs_ = new FeatureMatchPair [rhs.nmatch_];
        for(int i = 0; i < rhs.nmatch_; i++)
        {
            match_pairs_[i] = rhs.match_pairs_[i];
        }
        delete [] originMatchPairs;
        return *this;
    }

    // destructor
    ~SiftMatchPair()
    {
        if(match_pairs_ != NULL)
        {
            delete [] match_pairs_;
        }
    }
	
    // Helper functions
    inline const std::string & fileName1() const { return filename1_; }
    inline std::string & fileName1() { return filename1_; }
    inline const std::string & fileNmae2() const { return filename2_; }
    inline std::string & fileNmae2() { return filename2_; }
    inline const int & numMatches() const { return nmatch_; }
    inline int & numMatches() { return nmatch_; }
    inline const int & hInlierNumMatches() const { return homography_inlier_num_; }		// Homography inlier number of matches
    inline int & hInlierNumMatches() { return homography_inlier_num_; }
    inline const int & fInlierNumMatches() const { return fundamental_inlier_num_; }	// Fundamental matrix_ inlier number of matches
    inline int & fInlierNumMatches() { return fundamental_inlier_num_; }

    const FeatureMatchPair * matchPairs() const { return match_pairs_; }
    FeatureMatchPair * matchPairs() { return match_pairs_; }

    /// @brief WriteSiftMatchPair
    /// @param file is open with append mode
    /// @return
    bool WriteSiftMatchPair(FILE *file) const
    {
        // write information
        int filename2_length = filename2_.size();
        fwrite((void*)&filename2_length, sizeof(int), 1, file);
        fwrite((void*)filename2_.c_str(), sizeof(char), filename2_length, file);
        fwrite((void*)&nmatch_, sizeof(int), 1, file);
        fwrite((void*)&homography_inlier_num_, sizeof(int), 1, file);
        fwrite((void*)&fundamental_inlier_num_, sizeof(int), 1, file);

        fwrite((void*)homography_.rawVec(), sizeof(double), 9, file);
        fwrite((void*)fundamental_matrix_.rawVec(), sizeof(double), 9, file);

        // write match correspondence (int, int, char)
        fwrite((void*)match_pairs_, sizeof(FeatureMatchPair), nmatch_, file);
        return true;
    }

    /// @brief ReadSiftMatchPair reads from a file formatted with WriteSiftMatchPair, and then
    ///        initialize the SiftMatchPair class with the information extracted from the file
    ///
    /// @param file is open with append mode
    /// @return
    bool ReadSiftMatchPair(FILE *file)
    {
        char filename2_temp[256];
        int filename2_length;

        int check = fread((void*)&filename2_length, sizeof(int), 1, file);
        if(check != 1)  return false;

        fread((void*)filename2_temp, sizeof(char), filename2_length, file);
        filename2_temp[filename2_length] = '\0';
        filename2_ = std::string(filename2_temp);

        fread((void*)&nmatch_, sizeof(int), 1, file);
        fread((void*)&homography_inlier_num_, sizeof(int), 1, file);
        fread((void*)&fundamental_inlier_num_, sizeof(int), 1, file);

        fread((void*)homography_.rawVec(), sizeof(double), 9, file);
        fread((void*)fundamental_matrix_.rawVec(), sizeof(double), 9, file);

        // read match correspondence (int, int, char)
        if(match_pairs_ != NULL)
        {
            delete [] match_pairs_;
            match_pairs_ = NULL;
        }
        match_pairs_ = new FeatureMatchPair [nmatch_];
        fread((void*)match_pairs_, nmatch_, sizeof(FeatureMatchPair), file);
        return true;
    }

    void showInfo() const
    {
        zeta::notify(zeta::Notify::Gossip) << "filename1: " << filename1_ << '\n';
        zeta::notify(zeta::Notify::Gossip) << "filename2: " << filename2_ << '\n';
        zeta::notify(zeta::Notify::Gossip) << "nmatch: " << nmatch_ << '\n';
        zeta::notify(zeta::Notify::Gossip) << "homography_inlier_num: " << homography_inlier_num_ << '\n';
        zeta::notify(zeta::Notify::Gossip) << "fundamental_inlier_num: " << fundamental_inlier_num_ << '\n';
        zeta::notify(zeta::Notify::Gossip) << "homography: " << homography_ << '\n';
        zeta::notify(zeta::Notify::Gossip) << "fundamental_matrix: " << fundamental_matrix_ << '\n';

        // random shuffle
//        std::srand(unsigned (std::time(0)));
        std::vector<int> index_array(nmatch_);
//        std::iota(index_array.begin(), index_array.end(), 0);
        for (size_t i = 0; i < index_array.size(); ++i) index_array[i] = static_cast<int>(i);
        std::random_shuffle(index_array.begin(), index_array.end());

        for(size_t i = 0; i < 10; i++)
        {
            int index = index_array[i];
            match_pairs_[index].showInfo();
        }
    }
};

class SiftMatchFile
{
private:
    int image_num_;      // the number of match pairs
    std::string mat_filename_;   // the name of the .mat file
    std::vector<SiftMatchPair> match_pairs_;

public:
    SiftMatchFile()     // anonymous .mat file
    {
        image_num_ = 0;
    }
    SiftMatchFile(std::string mat_path)
    {
        mat_filename_ = mat_path;
        image_num_ = 0;
    }

    ///
    /// \brief ReadMatchFile: read a .mat file and save the matching infomation in a SiftMatchFile instance
    /// \param mat_path
    /// \return
    ///
    bool ReadMatchFile(std::string mat_path)
    {
        FILE *fd = fopen(mat_path.c_str(), "rb");
        if(fd == NULL)
        {
            zeta::notify(zeta::Notify::Error) << "Can't read .mat file. Exit...\n";
            return false;
        }
        SiftMatchPair temp_mp = SiftMatchPair(mat_path);
        while(temp_mp.ReadSiftMatchPair(fd))
        {
            match_pairs_.push_back(temp_mp);
        }
        image_num_ = match_pairs_.size();
        return true;
    }

    const int & getMatchNum() const { return image_num_; }
    const std::string & getMatFilename() const { return mat_filename_; }
    const std::vector<SiftMatchPair> & getSiftMatchPairs() const { return match_pairs_; }
};

//TODO(tianwei): copy assignment operator and copy constructor, using default is dangerous
class HomographyModel
{
public:
    // must-defined typedef for ransac framework
    typedef int Measurement;
    typedef zeta::V2d Data;
    typedef double DataIterator;

    inline Measurement worstMeasurement() const {return 0;}
    inline Measurement initialMeasurement() const {return 0;}
    inline int minimalDataNumber() const {return 4;}

    HomographyModel(int match_buf[][2], int nmatch, int npoint1, int npoint2, LTYPE *loc1, LTYPE *loc2, int loc_dim, double threshold)
    {
        match_buf_ = match_buf;
        nmatch_ = nmatch;
        npoint1_ = npoint1;
        npoint2_ = npoint2;
        loc1_ = loc1;
        loc2_ = loc2;
        loc_dim_ = loc_dim;
        threshold_ = threshold;
    }

    Measurement validateData(std::vector<bool> &flags)
    {
        int inlier_num = 0;
        for(size_t i = 0; i < nmatch_; i++)
        {
            flags[i] = false;
            assert(match_buf_[i][0] < npoint1_ && match_buf_[i][1] < npoint2_);

            double x = loc1_[match_buf_[i][0] * loc_dim_];
            double y = loc1_[match_buf_[i][0] * loc_dim_ + 1];

            double u = loc2_[match_buf_[i][1] * loc_dim_];
            double v = loc2_[match_buf_[i][1] * loc_dim_ + 1];

            zeta::V3d x1(x, y, 1);
            zeta::V3d x2(u, v, 1);
            zeta::V3d x1_trans = homography_matrix_ * x1;
            x1_trans = zeta::V3d(x1_trans[0]/x1_trans[2], x1_trans[1]/x1_trans[2], 1.0);

            double distance = sqrt((x2[0] - x1_trans[0]) * (x2[0] - x1_trans[0]) + (x2[1] - x1_trans[1]) * (x2[1] - x1_trans[1]));

            if(distance < threshold_)
            {
                flags[i] = true;
                inlier_num++;
            }
        }

        return inlier_num;
    }

    bool operator() (std::vector<int> const &indices)
    {
        if(indices.size() < minimalDataNumber())
        {
            return false;
        }
        zeta::Matd coefficient_mat;
        coefficient_mat.resize(2 * minimalDataNumber(), 9);

        // fill the matrix
        for(size_t i = 0; i < minimalDataNumber(); i++)
        {
            //(x, y, 1) vs. (u, v, 1)
            assert(match_buf_[indices[i]][0] < npoint1_ && match_buf_[indices[i]][1] < npoint2_);

            double x = loc1_[match_buf_[indices[i]][0] * loc_dim_];
            double y = loc1_[match_buf_[indices[i]][0] * loc_dim_ + 1];
            double z = 1.0;

            double u = loc2_[match_buf_[indices[i]][1] * loc_dim_];
            double v = loc2_[match_buf_[indices[i]][1] * loc_dim_ + 1];
            double w = 1.0;

            coefficient_mat(2*i, 0) = 0;
            coefficient_mat(2*i, 1) = 0;
            coefficient_mat(2*i, 2) = 0;
            coefficient_mat(2*i, 3) = - w * x;
            coefficient_mat(2*i, 4) = - w * y;
            coefficient_mat(2*i, 5) = - w * z;
            coefficient_mat(2*i, 6) = v * x;
            coefficient_mat(2*i, 7) = v * y;
            coefficient_mat(2*i, 8) = v * z;

            coefficient_mat(2*i+1, 0) = w * x;
            coefficient_mat(2*i+1, 1) = w * y;
            coefficient_mat(2*i+1, 2) = w * z;
            coefficient_mat(2*i+1, 3) = 0;
            coefficient_mat(2*i+1, 4) = 0;
            coefficient_mat(2*i+1, 5) = 0;
            coefficient_mat(2*i+1, 6) = - u * x;
            coefficient_mat(2*i+1, 7) = - u * y;
            coefficient_mat(2*i+1, 8) = - u * z;
        }

        zeta::SVD<zeta::Matd> svd(coefficient_mat);
        for(size_t row = 0; row < 3; row++)
        {
            for(size_t col = 0; col < 3; col++)
            {
                homography_matrix_(row, col) = svd.v()(3 * row + col, 8);
            }
        }
        return true;
    }

    const zeta::Mat3d &getHomography()
    {
        return homography_matrix_;
    }

private:
    zeta::Mat3d homography_matrix_;
    int (*match_buf_)[2];
    int nmatch_;
    int npoint1_;
    int npoint2_;
    LTYPE *loc1_;
    LTYPE *loc2_;
    int loc_dim_;
    double threshold_;
};

//TODO(tianwei): copy assignment operator and copy constructor, using default is dangerous
class FundamentalMatrixModel
{
public:
    // must-defined typedef for ransac framework
    typedef int Measurement;
    typedef zeta::V2d Data;
    typedef double DataIterator;

    inline Measurement worstMeasurement() const {return 0;}
    inline Measurement initialMeasurement() const {return 0;}
    inline int minimalDataNumber() const {return 8;}

    FundamentalMatrixModel(int match_buf[][2], int nmatch, int npoint1, int npoint2, LTYPE *loc1, LTYPE *loc2, int loc_dim, double threshold1, double threshold2)
    {
        match_buf_ = match_buf;
        nmatch_ = nmatch;
        npoint1_ = npoint1;
        npoint2_ = npoint2;
        loc1_ = loc1;
        loc2_ = loc2;
        loc_dim_ = loc_dim;
        threshold1_ = threshold1;
        threshold2_ = threshold2;
    }

    Measurement validateData(std::vector<bool> &flags)
    {
        int inlier_num = 0;
        for(size_t i = 0; i < nmatch_; i++)
        {
            flags[i] = false;
            assert(match_buf_[i][0] < npoint1_ && match_buf_[i][1] < npoint2_);
            double x = loc1_[match_buf_[i][0] * loc_dim_];
            double y = loc1_[match_buf_[i][0] * loc_dim_ + 1];

            double u = loc2_[match_buf_[i][1] * loc_dim_];
            double v = loc2_[match_buf_[i][1] * loc_dim_ + 1];

            zeta::V3d x1(x, y, 1);
            zeta::V3d x2(u, v, 1);
            zeta::V3d line1 = fundamental_matrix_.transpose() * x2;
            zeta::V3d line2 = fundamental_matrix_ * x1;
            line1 = zeta::V3d(line1[0]/line1[2], line1[1]/line1[2], 1.0);
            line2 = zeta::V3d(line2[0]/line2[2], line2[1]/line2[2], 1.0);
            double distance1 = abs(x1.dot(line1));
            double distance2 = abs(x2.dot(line2));

            if(abs(distance1) < threshold1_ && abs(distance2) < threshold2_)
            {
                flags[i] = true;
                inlier_num++;
            }
        }
        return inlier_num;
    }

    bool operator() (std::vector<int> const &indices)
    {
        if(indices.size() < minimalDataNumber())
        {
            return false;
        }
        zeta::Matd coefficient_mat;
        coefficient_mat.resize(minimalDataNumber(), 9);

        /// transform the points
        zeta::V3d center1(0.0, 0.0, 0.0), center2(0.0, 0.0, 0.0);
        std::vector<zeta::V3d> points1;
        std::vector<zeta::V3d> points2;
        zeta::Mat3d T1, T2;
        double s1, s2;          //scaling factors

        // translate the points so that the center is (0, 0, 1)
        for(size_t i = 0; i < minimalDataNumber(); i++)
        {
            assert(match_buf_[indices[i]][0] < npoint1_ && match_buf_[indices[i]][1] < npoint2_);

            double x = loc1_[match_buf_[indices[i]][0] * loc_dim_];
            double y = loc1_[match_buf_[indices[i]][0] * loc_dim_ + 1];
            points1.push_back(zeta::V3d(x, y, 1));
            center1 += points1[i];

            double u = loc2_[match_buf_[indices[i]][1] * loc_dim_];
            double v = loc2_[match_buf_[indices[i]][1] * loc_dim_ + 1];
            points2.push_back(zeta::V3d(u, v, 1));
            center2 += points2[i];
        }
        center1 /= minimalDataNumber();
        center2 /= minimalDataNumber();

        T1(0, 0) = 1;T1(0, 1) = 0;T1(0, 2) = -center1[0];
        T1(1, 0) = 0;T1(1, 1) = 1;T1(1, 2) = -center1[1];
        T1(2, 0) = 0;T1(2, 1) = 0;T1(2, 2) = 1;

        T2(0, 0) = 1;T2(0, 1) = 0;T2(0, 2) = -center2[0];
        T2(1, 0) = 0;T2(1, 1) = 1;T2(1, 2) = -center2[1];
        T2(2, 0) = 0;T2(2, 1) = 0;T2(2, 2) = 1;

        double dist_sum1 = 0, dist_sum2 = 0;
        for(size_t i = 0; i < minimalDataNumber(); i++)
        {
            zeta::V3d point1_translation = T1 * points1[i];
            zeta::V3d point2_translation = T2 * points2[i];
            point1_translation[2] = 0;      // force the third component to be 0
            point2_translation[2] = 0;      // in order to calculate the Euclidean distance

            dist_sum1 += point1_translation.norml2();
            dist_sum2 += point2_translation.norml2();
        }
        s1 = sqrt((double)2) * minimalDataNumber() / dist_sum1;
        s2 = sqrt((double)2) * minimalDataNumber() / dist_sum2;

        T1(0, 0) = s1; T1(0, 1) = 0; T1(0, 2) = -s1 * center1[0];
        T1(1, 0) = 0; T1(1, 1) = s1; T1(1, 2) = -s1 * center1[1];
        T1(2, 0) = 0; T1(2, 1) = 0; T1(2, 2) = 1;

        T2(0, 0) = s2; T2(0, 1) = 0; T2(0, 2) = -s2 * center2[0];
        T2(1, 0) = 0; T2(1, 1) = s2; T2(1, 2) = -s2 * center2[1];
        T2(2, 0) = 0; T2(2, 1) = 0; T2(2, 2) = 1;

        for(size_t i = 0; i < minimalDataNumber(); i++)
        {
            points1[i] = T1 * points1[i];
            points2[i] = T2 * points2[i];
        }

        // fill the matrix
        for(size_t i = 0; i < minimalDataNumber(); i++)
        {
            coefficient_mat(i, 0) = points2[i][0] * points1[i][0];
            coefficient_mat(i, 1) = points2[i][0] * points1[i][1];
            coefficient_mat(i, 2) = points2[i][0];
            coefficient_mat(i, 3) = points2[i][1] * points1[i][0];
            coefficient_mat(i, 4) = points2[i][1] * points1[i][1];
            coefficient_mat(i, 5) = points2[i][1];
            coefficient_mat(i, 6) = points1[i][0];
            coefficient_mat(i, 7) = points1[i][1];
            coefficient_mat(i, 8) = 1;
        }

        zeta::SVD<zeta::Matd> svd_coefficient_mat(coefficient_mat);
        for(size_t row = 0; row < 3; row++)
        {
            for(size_t col = 0; col < 3; col++)
            {
                fundamental_matrix_(row, col) = svd_coefficient_mat.v()(3 * row + col, 8);
            }
        }

        // enforcing the rank two constraint by setting the smallest singular value of F to be zero
        zeta::SVD<zeta::Mat3d> svd_fundamental_matrix(fundamental_matrix_);
        zeta::Mat3d fundamental_matrix_u;
        zeta::Mat3d fundamental_matrix_w;
        zeta::Mat3d fundamental_matrix_v;

        // TODO(tianwei): we can't multiply, without copy, figure out why
        fundamental_matrix_u = svd_fundamental_matrix.u();

        // recomposing the matrix W, enforcing the rank 2 constraint
        for(size_t i = 0; i < 3; i++)
        {
            for(size_t j = 0; j < 3; j++)
            {
                if(i == j)
                {
                    fundamental_matrix_w(i, j) = svd_fundamental_matrix.w()[i];
                }
                else
                {
                    fundamental_matrix_w(i, j) = 0;
                }
            }
        }
        fundamental_matrix_w(2, 2) = 0;

        // TODO(tianwei): we can't multiply, without copy, figure out why
        fundamental_matrix_v = svd_fundamental_matrix.v();

        fundamental_matrix_ = fundamental_matrix_u.mul(fundamental_matrix_w).mul(fundamental_matrix_v.transpose());

        return true;
    }

    const zeta::Mat3d &getFundmantalMatrix(){ return fundamental_matrix_; }

private:
    zeta::Mat3d fundamental_matrix_;
    int (*match_buf_)[2];
    int nmatch_;
    int npoint1_;
    int npoint2_;
    LTYPE *loc1_;
    LTYPE *loc2_;
    int loc_dim_;
    double threshold1_;
    double threshold2_;
};

}   // end of namespace tw

#endif // MATCHHANDLER_H
