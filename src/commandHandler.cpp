#include "commandHandler.h"

#include <vector>
#include <string>
#include <Base/Common/ioutilities.h>
#include <Base/Common/serializer.h>
#include <Base/Common/notify.h>
#include <Base/Common/stopwatch.h>
#include <Base/Common/cmdframework.h>
#include <Base/image/pnm.h>
#include <Base/misc/imagehelper.h>

bool GlslFeatureHandler(std::vector<std::string> & commandOptions);
bool GlslBatchFeatureHandler(std::vector<std::string> & commandOptions);
bool SfmBatchFeatureHandler(std::vector<std::string> & commandOptions);
bool GlslMatchHandler(std::vector<std::string> & commandOptions);
bool SequenceMatchHandler(std::vector<std::string> & commandOptions);
bool ShowMatchHandler(std::vector<std::string> & commandOptions);

bool commandHandler(std::vector<std::string> & commandOptions)
{
    zeta::CmdFramework cmd_framework("sift");

    cmd_framework.addCommand("glsl",
                             "SiftGPU, using GLSL",
                             zeta::CmdFramework::CommandFn(GlslFeatureHandler, NULL));

    cmd_framework.addCommand("glslb",
                             "SiftGPU, GLSL batch version, input is a image list",
                             zeta::CmdFramework::CommandFn(GlslBatchFeatureHandler, NULL));

    cmd_framework.addCommand("sfm",
                             "SiftGPU, VisualSFM batch version, input is a image list",
                             zeta::CmdFramework::CommandFn(SfmBatchFeatureHandler, NULL));

    cmd_framework.addCommand("match",
                             "Matching module for SiftGPU",
                             zeta::CmdFramework::CommandFn(GlslMatchHandler, NULL));

    cmd_framework.addCommand("sequence",
                             "Sequence matching module for SiftGPU",
                             zeta::CmdFramework::CommandFn(SequenceMatchHandler, NULL));

    cmd_framework.addCommand("show",
                             "Show matching result",
                             zeta::CmdFramework::CommandFn(ShowMatchHandler, NULL));

    return (cmd_framework.exe(commandOptions) == 0);
}

void commandHelp()
{
    std::cerr << "This command is to compute and match sift.\n"
              << "================================================\n"
              << "Usage: -sift glsl <image_path> [--output_path \\path\\of\\siftfile]\n"
              << "Usage: -sift glslb <image_list_file> [--output_path \\path\\to\\siftfile\\dir]\n"
              << "Usage: -sift sfm <image_list_file> [--output_path \\path\\to\\siftfile\\dir]\n"
              << "Usage: -sift match <sift_list_file> [--output_path \\path\\to\\matfile\\dir]\n"
              << "Usage: -sift sequence <image_list_file> \n";
}
