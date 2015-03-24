#ifndef FEATUREHANDLER_H
#define FEATUREHANDLER_H

#include <vector>
#include <string>
#include <siftgpu/SiftGPU.h>
#include <siftgpu/SiftPyramid.h>
#include <siftgpu/GlobalUtil.h>
#include <siftgpu/FeaturePoints.h>

#include <Base/Common/notify.h>

/*
    // Library or binary for feature detection
    // 0: CPU (depending on param_use_vlfeat_or_lowe),
    // 1: GLSL-based SiftGPU, 2 CUDA-based SiftPGU,
    // 3: customized SiftGPU (depending on param_use_siftgpu_customize)
    size_t use_sift_category;

    // Device for feature matching
    // 0: cpu, 1: glsl, 2+: cuda if compiled in SiftGPU.dll
    size_t use_sift_match_category;
    size_t cpu_sift_match_num_thread;

    // Disable GPU-based computation when remote desktop is detected
    // Change this to 0 if VisualSFM incorrectly detects remote desktop
    size_t check_remote_desktop;

    // Feature matching parameters
    size_t gpu_match_fmax;          // Max features used in gpu sift matching
    size_t sift_max_dist;           // Max sift matching distance
    size_t sift_max_dist_ratio;     // Max sift matching distance ratio
    size_t sift_match_mbm;    // Mutual best match or not
    size_t no_stationary_points;    // Filter the stationary point matching
*/

struct FeatureParam {
    // necessary parameters
    std::string input_path;

    // optional parameters
    std::string output_path;
    short int verbose;              // verbose level for output information
    bool is_binary_sift;            // whether set binary sift
    int maxd;                       // max working dimension
    int cuda_device;                // device num for cuda(multi-GPU)

    FeatureParam()
    {
        verbose = 0;
        is_binary_sift = false;
    }
};

namespace tw {
class SiftData
{
private:
    int name_;
    int version_;
    int npoint_;
    int nLocDim_;
    int nDesDim_;
    DTYPE *dp_;
    LTYPE *lp_;

public:
    SiftData()
    {
        name_ = 'S' + ('I' << 8) + ('F' << 16) + ('T' << 24);
        version_ = 'V' + ('5' << 8) + ('.' << 16) + ('0' << 24);
        dp_ = NULL; lp_ = NULL;
    }

    // copy constructor
    SiftData(const SiftData & obj):
        name_(obj.name_), version_(obj.version_), npoint_(obj.npoint_),
        nLocDim_(obj.nLocDim_), nDesDim_(obj.nDesDim_)
    {
        //copy feature descriptors
        DTYPE *dp_origin = dp_;
        dp_ = new DTYPE [obj.npoint_];
        memcpy(dp_, obj.dp_, sizeof(DTYPE)*obj.npoint_);
        delete [] dp_origin;

        //copy location descriptors
        LTYPE *lp_origin = lp_;
        lp_ = new LTYPE [obj.npoint_];
        memcpy(lp_, obj.lp_, sizeof(LTYPE)*obj.npoint_);
        delete [] lp_origin;
    }

    // copy assignment operator
    SiftData & operator=(const SiftData &rhs)
    {
        name_ = rhs.name_;
        version_ = rhs.version_;
        npoint_ = rhs.npoint_;
        nLocDim_ = rhs.nLocDim_;
        nDesDim_ = rhs.nDesDim_;

        //copy feature descriptors
        DTYPE *dp_origin = dp_;
        dp_ = new DTYPE [rhs.npoint_];
        memcpy(dp_, rhs.dp_, sizeof(DTYPE)*rhs.npoint_);
        delete [] dp_origin;

        //copy location descriptors
        LTYPE *lp_origin = lp_;
        lp_ = new LTYPE [rhs.npoint_];
        memcpy(lp_, rhs.lp_, sizeof(LTYPE)*rhs.npoint_);
        delete [] lp_origin;

        return *this;
    }

    // destructor
    ~SiftData()
    {
        if (NULL != dp_) delete [] dp_;
        if (NULL != lp_) delete [] lp_;
    }

    ///
    /// \brief ReadSiftFile: read sift file, this function is compatible with the sfm version of sift file
    /// \param szFileName: the name of the sift file
    /// \return
    ///
    bool ReadSiftFile(std::string const &szFileName)
    {
        name_ = 'S' + ('I' << 8) + ('F' << 16) + ('T' << 24);
        version_ = 'V' + ('5' << 8) + ('.' << 16) + ('0' << 24);
        FILE *fd;
        if((fd = fopen(szFileName.c_str(), "rb")) == NULL)
        {
            zeta::notify(zeta::Notify::Gossip) << "Can't read sift file " << szFileName << '\n';
            return false;
        }
        fread(&name_, sizeof(int), 1, fd);
        fread(&version_, sizeof(int), 1, fd);

        if(name_ == ('S'+ ('I'<<8)+('F'<<16)+('T'<<24)))
        {
            fread(&npoint_, sizeof(int), 1, fd);
            fread(&nLocDim_, sizeof(int), 1, fd);
            fread(&nDesDim_, sizeof(int), 1, fd);

            lp_ = new LTYPE [npoint_ * nLocDim_];          //restoring location data
            dp_ = new DTYPE [npoint_ * nDesDim_];          //restoring descriptor data
            if(npoint_ > 0 && nLocDim_ > 0 && nDesDim_ == 128)
            {
                fread(lp_, sizeof(LTYPE), nLocDim_ * npoint_, fd);
                fread(dp_, sizeof(DTYPE), nDesDim_ * npoint_, fd);
                // read the eof sign, here we just ignore it
                //fread(&sift_eof, sizeof(int), 1, fd);     //in the sfm save format, there is a EOF sign at last
                fclose(fd);
            }
            else
            {
                fclose(fd);
                return false;
            }

        }
        else
        {
            fclose(fd);
            return false;
        }
        return true;
    }

    int getFeatureNum() {return npoint_;}
    int getLocDim() {return nLocDim_;}
    int getDesDim() {return nDesDim_;}
    LTYPE * getLocPointer() {return lp_;}
    DTYPE * getDesPointer() {return dp_;}
};

}

#endif // SIFTHANDLER_H
