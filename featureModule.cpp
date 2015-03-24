#include "featureModule.h"

#include <Base/Common/notify.h>
#include <Base/geometry/bdlgeometry.h>
#include <Base/misc/imagehelper.h>
#include <Base/misc/phototourismhelper.h>
#include <Base/Common/cmdframework.h>
#include <Base/Common/ioutilities.h>
#include <Base/misc/formatconvertor.h>
#include <Base/Common/stopwatch.h>

#include <cstdlib>
#include <vector>
#include <iostream>
#include "GL/glew.h"

using std::cout;
using std::endl;
////////////////////////////////////////////////////////////////////////////
#if !defined(SIFTGPU_STATIC) && !defined(SIFTGPU_DLL_RUNTIME)
// SIFTGPU_STATIC comes from compiler
#define SIFTGPU_DLL_RUNTIME
// Load at runtime if the above macro defined
// comment the macro above to use static linking
#endif

////////////////////////////////////////////////////////////////////////////
// define REMOTE_SIFTGPU to run computation in multi-process (Or remote) mode
// in order to run on a remote machine, you need to start the server manually
// This mode allows you use Multi-GPUs by creating multiple servers
// #define REMOTE_SIFTGPU
// #define REMOTE_SERVER        NULL
// #define REMOTE_SERVER_PORT   7777


///////////////////////////////////////////////////////////////////////////
//#define DEBUG_SIFTGPU  //define this to use the debug version in windows

#ifdef _WIN32
#ifdef SIFTGPU_DLL_RUNTIME
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define FREE_MYLIB FreeLibraREMOTE_SIFTGPUry
#define GET_MYPROC GetProcAddress
#else
//define this to get dll import definition for win32
#define SIFTGPU_DLL
#ifdef _DEBUG
#pragma comment(lib, "../../lib/siftgpu_d.lib")
#else
#pragma comment(lib, "../../lib/siftgpu.lib")
#endif
#endif
#else
#ifdef SIFTGPU_DLL_RUNTIME
#include <dlfcn.h>
#define FREE_MYLIB dlclose
#define GET_MYPROC dlsym
#endif
#endif

namespace tw
{

bool SiftGPUParseParam(FeatureParam &feature_param, std::vector<std::string> &commandOptions)
{
    zeta::OptionParser parser;
    if(!parser.registerOption("input_path", "", "input path")
            || !parser.registerOption("--output_path", "", "optional output path for .sift files")
            || !parser.registerOption("--verbose", "", "output verbose level")
            || !parser.registerOption("--maxd", "3200", "the maximum working dimension")
            || !parser.registerOnOffOption("--binary_sift", "use binary format to output the sift file")
            || !parser.registerOption("--cuda", "-1", "cuda device num (for one GPU just use 0)"))
    {
        return false;
    }

    if (!parser.parse(commandOptions))
    {
        return false;
    }

    feature_param.input_path = parser.getString("input_path");
    feature_param.output_path = parser.getString("--output_path");

    feature_param.is_binary_sift = parser.getValue<bool>("--binary_sift");
    feature_param.maxd = parser.getValue<int>("--maxd");
    feature_param.verbose = parser.getValue<short int>("--verbose");
    feature_param.cuda_device = parser.getValue<int>("--cuda");

    return true;
}

bool InitSiftGPU(SiftGPU & sift, const FeatureParam & feature_param)
{
    char *argv[] = { "-fo", "-1", "-tc2", "7680", "-nomc"};
    //-fo -1, starting from -1 octave
    //-v 1, only print out # feature and overall time
    //-nomc disable auto-downsampling that try to fit GPU memory cap
    int argc = sizeof(argv)/sizeof(char*);
    sift.ParseParam(argc, argv);
    sift.SetVerbose(feature_param.verbose);
    GlobalUtil::_BinarySIFT = feature_param.is_binary_sift; //sift.SetBinarySift(feature_param.is_binary_sift);
    GlobalUtil::_texMaxDim = feature_param.maxd > 0 ? feature_param.maxd : 3200;        // set max dimension, default is 3200
    if(feature_param.cuda_device != -1 && feature_param.cuda_device >= 0)               // enable cuda
    {
        GlobalUtil::_UseCUDA = 1;
        GlobalParam::_DeviceIndex = feature_param.cuda_device;
    }

    if(sift.CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED)
    {
        return false;
    }
    return true;
}

///
/// \brief GetSiftPathFromImageFile: compose sift file path (.sift) from image file (.jpg, .pgm, ...)
/// \param image_path: the image file path
/// \param is_output_path: whether we specify the --output_path parameter
/// \param output_path: the specified output directory
/// \return
///
std::string GetSiftPathFromImageFile(const std::string & image_path, bool is_output_path, const std::string & output_path = "")
{
    std::string filename_without_suffix = IOUtilities::SplitPathExt(image_path).first;
    std::string output_siftfile;
    if(is_output_path)      // output to the assigned folder
    {
        filename_without_suffix = IOUtilities::SplitPath(filename_without_suffix).second;
        output_siftfile = IOUtilities::JoinPath(output_path, filename_without_suffix) + ".sift";
    }
    else        // output to the same folder
    {
        output_siftfile = filename_without_suffix + ".sift";
    }
    return output_siftfile;
}

}   // end of namespace tw


/** @brief Atomic write for SiftGPU - VisualSFM version. The output of the standard version of SiftGPU
 * is not suitable for VisualSFM. But actually the two softwares are both implemented by Changchang Wu.
 * Don't know what he was thinking about...
 */
bool SiftSfmAtomicWrite(const char *szFile, SiftGPU &sift, unsigned char * &image_data, int image_width, int image_height, int num_channel)
{
    struct sift_fileheader_v2
    {
        int	 szFeature;
        int  szVersion;
        int  npoint;
        int  nLocDim;
        int  nDesDim;
    }sfh;

    // write header information
    int i,j, sift_eof = (0xff+('E'<<8)+('O'<<16)+('F'<<24));
    FILE * fd;
    zeta::AtomicWriteHelper<FILE *> helper(std::string(szFile), fd);
    if(fd == NULL) return false;

    sfh.szFeature = ('S' + ('I'<<8) + ('F'<<16) + ('T'<<24));
    if(num_channel == 3)
    {
        sfh.szVersion = ('V'+('5'<<8)+('.'<<16)+('0'<<24));
    }
    else    //num_channel = 1
    {
        sfh.szVersion = ('V' + ('4' << 8) + ('.' << 16) + ('0' << 24));
    }
    sfh.npoint  = sift.GetFeatureNum();
    sfh.nLocDim = 5;
    sfh.nDesDim = 128;
    fwrite(&sfh, sizeof(sfh), 1, fd);

    float * lp;
    float * dp;
    unsigned char* fph;
    float  *fp;
    unsigned char * ucp;
    int Max, MemNum;

    lp = &sift.GetSiftPyramid()->_keypoint_buffer[0];

    MemNum = sfh.nDesDim * sfh.npoint;
    Max = sfh.npoint*sfh.nLocDim;
    fph= new unsigned char[MemNum];
    fp = (float*) fph;

    //location
    unsigned char r, g, b;
    for(i = 0; i < sfh.npoint; i++, lp+=4)
    {
        *fp++ = (float) *lp;            //x
        *fp++ = (float) *(lp+1);        //y

        if(num_channel == 3)
        {
            int x,y;
            x = (int)*lp;
            y = (int)*(lp+1);
            int index = (y * image_width + x) * num_channel;
            memcpy(fp, (image_data + index), num_channel * sizeof(unsigned char));
            fp++;
        }
        else
        {
            *fp++ = 0;
        }
        *fp++ = (float) *(lp+2);        //scale
        *fp++ = (float) *(lp+3);        //orientation
    }

    fwrite(fph, sizeof(float)*Max, 1, fd);

    //descriptor
    dp = &sift.GetSiftPyramid()->_descriptor_buffer[0];

    Max = sfh.npoint*sfh.nDesDim;
    ucp = (unsigned char*) fph;
    for( i = 0; i < sfh.npoint; i++)
    {
        for(j = 0; j < sfh.nDesDim; j++, dp++)
        {
            *ucp++ = ((unsigned int)floor(0.5+512.0f*(*dp)));
        }
    }
    fwrite(fph, sizeof(unsigned char)*Max, 1, fd);
    fwrite(&sift_eof, sizeof(int), 1, fd);

    delete [] fph;

    return true;
}

/** @brief After extract the sift features, atomic write ensure that the written-to-file process is all or nothing
 *  @param output_path: the output path of the sift file.
 */
bool SiftAtomicWrite(std::string const &szFileName, SiftGPU &sift)
{
    int _featureNum = sift.GetFeatureNum();
    if (_featureNum <=0) return false;
    float * pk = &(sift.GetSiftPyramid()->_keypoint_buffer[0]);

    if(GlobalUtil::_BinarySIFT)
    {
        std::ofstream out;
        zeta::AtomicWriteHelper<std::ofstream> atomic_out(szFileName, out);

        out.write((char* )(&_featureNum), sizeof(int));

        if(GlobalUtil::_DescriptorPPT)
        {
            int dim = 128;
            out.write((char* )(&dim), sizeof(int));
            float * pd = &(sift.GetSiftPyramid()->_descriptor_buffer[0]) ;
            for(int i = 0; i < _featureNum; i++, pk+=4, pd +=128)
            {
                out.write((char* )(pk +1), sizeof(float));
                out.write((char* )(pk), sizeof(float));
                out.write((char* )(pk+2), 2 * sizeof(float));
                out.write((char* )(pd), 128 * sizeof(float));
            }
        }
        else
        {
            int dim = 0;
            out.write((char* )(&dim), sizeof(int));
            for(int i = 0; i < _featureNum; i++, pk+=4)
            {
                out.write((char* )(pk +1), sizeof(float));
                out.write((char* )(pk), sizeof(float));
                out.write((char* )(pk+2), 2 * sizeof(float));
            }
        }
        out.close();
    }
    else
    {
        float temp;
        std::ofstream out;
        out.flags(std::ios::fixed);
        zeta::AtomicWriteHelper<std::ofstream> atomic_out(szFileName, out);

        if(GlobalUtil::_DescriptorPPT)
        {
            float * pd = &(sift.GetSiftPyramid()->_descriptor_buffer[0]);
            out<<_featureNum<<" 128"<<std::endl;

            for(int i = 0; i < _featureNum; i++)
            {
                //in y, x, scale, orientation order
                out<<std::setprecision(2) << pk[1]<<" "<<std::setprecision(2) << pk[0]<<" "
                  <<std::setprecision(3) << pk[2]<<" " <<std::setprecision(3) <<  pk[3]<< std::endl;

                pk+=4;
                for(int k = 0; k < 128; k ++, pd++)
                {
                    if(GlobalUtil::_NormalizedSIFT)
                    {
                        if(temp < *pd){
                            temp = *pd;
                        }
                        out<< ((unsigned int)floor(0.5+512.0f*(*pd))) << " ";
                    }
                    else
                        out << std::setprecision(8) << pd[0] << " ";

                    if ( (k+1)%20 == 0 ) out<<std::endl;

                }
                out<<std::endl;

            }
        }
        else
        {
            out<<_featureNum<<" 0"<<std::endl;
            for(int i = 0; i < _featureNum; i++, pk+=4)
            {
                out<<pk[1]<<" "<<pk[0]<<" "<<pk[2]<<" " << pk[3]<<std::endl;
            }
        }
        out.close();
    }

    return true;
}

bool CudaFeatureHandler(std::vector<std::string> & commandOptions)
{
    return true;
}


/**
 * @brief GlslFeatureHandler takes a set of parameters and extract and save sift feature from an image
 * @param commandOptions: a set of input parameters including input/output path, optional parameters
 * @return
 */
// SiftGPU runs on GLSL by default, can't use multiple GPU
bool GlslFeatureHandler(std::vector<std::string> & commandOptions)
{
    FeatureParam feature_param;

    if(!tw::SiftGPUParseParam(feature_param, commandOptions))
    {
        zeta::notify(zeta::Notify::Error) << "Error: Fail to parse SiftGPU parameters.\n";
        return false;
    }
    std::string input_image_path = feature_param.input_path;
    std::string output_sift_path = feature_param.output_path;
    if(!IOUtilities::FileExist(input_image_path))
    {
        zeta::notify(zeta::Notify::Error) << "Image file doesn't exist.\n";
        return false;
    }

    if(IOUtilities::IsEmptyString(output_sift_path))    // if no output path
    {
        output_sift_path = IOUtilities::SplitPathExt(input_image_path).first + ".sift";
    }

    // initialize sift and set parameters
    SiftGPU sift;
    if(!tw::InitSiftGPU(sift, feature_param))
    {
        zeta::notify(zeta::Notify::Error) << "Initialize SiftGPU failed. Exit...\n";
        return false;
    }

    if(IOUtilities::FileExist(input_image_path))
    {
        zeta::Imageu input_image;
        zeta::loadImage(input_image, input_image_path);
        unsigned char *data = input_image.GetRawMemory();

        int image_width = input_image.GetWidth();
        int image_height = input_image.GetHeight();
        int num_channel = input_image.GetChannel();

        if(num_channel == 3)       //rgb images, convert to greyscale
        {
            int num_pixel = image_width * image_height;
            int i = 0, j = 0;
            for(; i < num_pixel; i++, j+=3)
            {
                data[i] = int(0.10454f* data[j+2]+0.60581f* data[j+1]+0.28965f* data[j]);
            }
        }

        if(sift.RunSIFT(input_image.GetWidth(), input_image.GetHeight(), data, GL_LUMINANCE, GL_UNSIGNED_BYTE))
        {
            SiftAtomicWrite(output_sift_path, sift);

            int num_features = sift.GetFeatureNum();
            zeta::notify(zeta::Notify::Gossip) << "[GPU Sift] Detect features in file: " << input_image_path << '\n'
                                               << "# of features: " << num_features << '\n';
        }
        else
        {
            zeta::notify(zeta::Notify::Error) << "Error [GPU Sift] Failt to detect features in " << input_image_path << '\n';
            return false;
        }
    }
    else
    {
        zeta::notify(zeta::Notify::Error) << "Error [CudaFeature] The input image " << input_image_path << " does not exist.\n";
        return false;
    }

    return true;
}

/**
 * @brief GlslBatchFeatureHandler is a batch version of GlslFeatureHandler, which takes a list of images
 * @param commandOptions: a set of input parameters including the path to the image list and optional parameters
 * @return
 */
/* The allocation is a time-consuming step. The best performance can be obtained
 * by pre-resize all images to a same size and process them with on SiftGPU instance.
 * The batch version takes a image list and extract the feature descriptors of these images
 * in a batch
 */
bool GlslBatchFeatureHandler(std::vector<std::string> & commandOptions)
{
    FeatureParam feature_param;

    if(!tw::SiftGPUParseParam(feature_param, commandOptions))
    {
        zeta::notify(zeta::Notify::Error) << "Error: Fail to parse SiftGPU parameters.\n";
        return false;
    }
    std::string image_list_file = feature_param.input_path;
    std::string sift_save_dir = feature_param.output_path;

    bool is_output_path = false;
    if(!IOUtilities::IsEmptyString(sift_save_dir))
    {
        is_output_path = true;
        IOUtilities::Mkdir(sift_save_dir);
    }

    // initialize sift and set parameters
    SiftGPU sift;
    if(!tw::InitSiftGPU(sift, feature_param))
    {
        zeta::notify(zeta::Notify::Error) << "Initialize SiftGPU failed. Exit...\n";
        return false;
    }

    std::vector<std::string> image_filenames;
    IOUtilities::ExtractNonEmptyLines(image_list_file, image_filenames);
    for(int i = 0; i < image_filenames.size(); i++){
        if(IOUtilities::FileExist(image_filenames[i])){
            std::string output_siftfile = tw::GetSiftPathFromImageFile(image_filenames[i], is_output_path, sift_save_dir);

            zeta::Imageu input_image;
            zeta::loadImage(input_image, image_filenames[i]);
            unsigned char *data = input_image.GetRawMemory();

            int image_width = input_image.GetWidth();
            int image_height = input_image.GetHeight();
            int num_channel = input_image.GetChannel();

            if(num_channel == 3)       //rgb images, convert to greyscale
            {
                int num_pixel = image_width * image_height;
                int i = 0, j = 0;
                for(; i < num_pixel; i++, j+=3)
                {
                    data[i] = int(0.10454f* data[j+2]+0.60581f* data[j+1]+0.28965f* data[j]);
                }
            }

            if(sift.RunSIFT(input_image.GetWidth(), input_image.GetHeight(), data, GL_LUMINANCE, GL_UNSIGNED_BYTE))
            {
                SiftAtomicWrite(output_siftfile, sift);
                int num_features = sift.GetFeatureNum();
                zeta::notify(zeta::Notify::Gossip) << "[GPU Sift] Detect features in file: " << image_filenames[i] << '\n'
                                                   << "# of features: " << num_features << '\n';
            }
            else
            {
                zeta::notify(zeta::Notify::Error) << "Error [GPU Sift] Failt to detect features in " << image_filenames[i] << '\n';
                return false;
            }
        }
        else
        {
            zeta::notify(zeta::Notify::Error) << "Error [CudaFeature] The input image " << image_filenames[i] << " does not exist.\n";
            return false;
        }
    }
    return true;
}


/**
 * @brief SfmBatchFeatureHandler creates sift file used by VisualSFM
 * @param commandOptions
 * @return
 */
bool SfmBatchFeatureHandler(std::vector<std::string> & commandOptions)
{
    zeta::StopWatches timers;
    FeatureParam feature_param;

    if(!tw::SiftGPUParseParam(feature_param, commandOptions))
    {
        zeta::notify(zeta::Notify::Error) << "Error: Fail to parse SiftGPU parameters.\n";
        return false;
    }
    std::string image_list_file = feature_param.input_path;
    std::string sift_save_dir = feature_param.output_path;

    bool is_output_path = false;
    if(!IOUtilities::IsEmptyString(sift_save_dir))
    {
        is_output_path = true;
        IOUtilities::Mkdir(sift_save_dir);
    }

    // initialize sift and set parameters
    SiftGPU sift;
    {
        zeta::LocalWatch timer("Config Sift", timers);
        if(!tw::InitSiftGPU(sift, feature_param))
        {
            zeta::notify(zeta::Notify::Error) << "Initialize SiftGPU failed. Exit...\n";
            return false;
        }
    }

    std::vector<std::string> image_filenames;
    IOUtilities::ExtractNonEmptyLines(image_list_file, image_filenames);

    zeta::StopWatch sift_time;
    sift_time.Start();
    for(int i = 0; i < image_filenames.size(); i++)
    {
        if(IOUtilities::FileExist(image_filenames[i]))
        {
            std::string output_siftfile = tw::GetSiftPathFromImageFile(image_filenames[i], is_output_path, sift_save_dir);
            bool run_sift_flag = true;

            if(IOUtilities::FileExist(output_siftfile))
            {
                tw::SiftData sift_data;
                if(sift_data.ReadSiftFile(output_siftfile))    // The sift file is valid, skip running feature detection
                {
                    run_sift_flag = false;
                    size_t num_features = sift_data.getFeatureNum();
                    zeta::notify(zeta::Notify::Gossip) << "[GPU Sift] Find existing feature file: " << output_siftfile << '\n'
                                                       << "# of features: " << num_features << '\n';
                }
            }

            if(run_sift_flag)
            {
                zeta::Imageu input_image;
                int image_width, image_height, num_channel;
                unsigned char *data = NULL;
                {
                    zeta::LocalWatch timer("Read Image", timers);
                    zeta::loadImage(input_image, image_filenames[i]);
                    data = input_image.GetRawMemory();

                    image_width = input_image.GetWidth();
                    image_height = input_image.GetHeight();
                    num_channel = input_image.GetChannel();

                    if(num_channel == 3)       //rgb images, convert to greyscale
                    {
                        int num_pixel = image_width * image_height;
                        int i = 0, j = 0;
                        for(; i < num_pixel; i++, j+=3)
                        {
                            data[i] = int(0.10454f* data[j+2]+0.60581f* data[j+1]+0.28965f* data[j]);
                        }
                    }
                }

                bool runsift_results = false;
                double runsift_elapse = 0;
                {
                    zeta::LocalWatch timer("Sift", timers);
                    runsift_results = sift.RunSIFT(input_image.GetWidth(), input_image.GetHeight(), data, GL_LUMINANCE, GL_UNSIGNED_BYTE);
                    runsift_elapse = zeta::StopWatch::ToSeconds(timer.elapsed());
                }

                if(runsift_results)
                {
                    zeta::LocalWatch timer("Save Sift", timers);
                    SiftSfmAtomicWrite(output_siftfile.c_str(), sift, data, image_width, image_height, num_channel);
                    int num_features = sift.GetFeatureNum();
                    zeta::notify(zeta::Notify::Gossip) << "[GPU Sift] Detect features in file: " << image_filenames[i] << '\n'
                                                       << "# of features: " << num_features << '\n'
                                                       << runsift_elapse << " sec\n\n";
                }
                else
                {
                    zeta::notify(zeta::Notify::Error) << "Error [GPU Sift] Faild to detect features in " << image_filenames[i] << '\n';
                    return false;
                }
            }
        }
        else
        {
            zeta::notify(zeta::Notify::Error) << "Error [CudaFeature] The input image " << image_filenames[i] << " does not exist.\n";
            return false;
        }
     }

    zeta::notify(zeta::Notify::Gossip) << timers << '\n';
    return true;
}

bool CpuFeatureHandler(std::vector<std::string> & commandOptions)
{
    return true;
}

bool VlFeatureHandler(std::vector<std::string> & commandOptions)
{
    return true;
}
