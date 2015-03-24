# Handler for SIFT #

## Notice ##
This is the showcase code for my application of GSoC 2015. Notice that this repository relies on SiftGPU by Changchang Wu (http://cs.unc.edu/~ccwu/siftgpu/) and our unshared repositories owned by our group, currently you can not build it successfully on your machine.

This README file documents how to use the handler to siftgpu. This mainly includes two parts: one for generating sift files, another for matching sift files. The following is the detailed descriptions.

## 1. Sift Files Generation ##

### Quick summary ###
This function generates sift files from image files. Currently, we support .jpg, .pgm, and .ppm format.

### Command ###
~~~~
./xms.sh -sift glsl <image_path> [--output_path \path\of\siftfile <other_optional_flags>]
~~~~

This bash command processes a image and output the sift file for this image. If the __--output_path__ is not specified, it will save the sift file to the same directory as the image file.

~~~~
./xms.sh -sift glslb <image_list_file> [--output_path \path\to\siftfile\dir <other_optional_flags>]
~~~~
This bash command is a batch version of the above command. It takes in a image list which contains all the absolute paths to the image files (ensure there is no repetition in the image_list since the sift handler doesn't have strong robustness at this moment). Users can specifies a saving directory for the sift files. If the __--output_path__ is not specified, it will save the sift files to the same directory as the image files.

~~~~
./xms.sh -sift sfm <image_list_file> [--output_path \path\to\siftfile\dir <other_optional_flags>]
~~~~
This bash command is also a batch version of sift files generation. It is different from the above __glslb__ version in that it saves sift file in a different format, which is in accord with *VisualSFM* input. __Note that currently this is the bash version you should use__, since the matching program process this type of sift files. As before, if the __--output_path__ is not specified, it will save the sift files to the same directory as the image files.

For other optional flags(__other_optional_flags__), please refer to the following list:

Optional flags  | Meaning
------------- | -------------
--output_path <string> | output .sift file to a specified directory
--verbose <int=0>   | verbose level of terminal output 
--binary_sift  | save the sift file in binary format
--maxd <int>   | specify maximum working dimension
--cuda <int>   | using cuda instead of GLSL

## 2. Matching File Generation ##

### Quick summary ###
This function first reads all (or partial) the sift files into the memory. Then it do a pairwise sift file matching to get the putative matches of feature points. It then calculates the fundamental matrix and homography for a image pair. It then outputs a match file for each image. Note that the match files are saved in a decremental  style. For example, if the image list contains *n* images, the match file of the first image contains the matching information between the first and the other *n-1* images. The matching file of the second image contains the matching information between the second image and the following *n-2* images, so on and so forth.
### Command ###

~~~~
./xms.sh -sift match <sift_list_file> [--output_path \path\to\matfile\dir <other_optional_flags>]
~~~~
This bash command takes in a sift-list-file and do a pairwise matching (ensure that the sift-list-file has no repetition). If the __--output_path__ is not specified, it will save the matching files to the same directory as the sift files.

For other optional flags(__other_optional_flags__), please refer to the following list:

Optional flags  | Meaning
------------- | -------------
--output_path <string> | output .mat file to a specified directory
--optional_match <string=\path\to\match\list>   | match specified image pairs
--cuda <int=0> | using CUDA to match feature points (this should be slightly faster than GLSL matching)
--image_list <string=\path\to\image\list> | used for computation of threhold


###The logic format of matching file###
Here we assume that there are *n* sift files in the sift-file-list, take the *i-th* sift file in the sift-file-list as an example, it contains *n - i* logic components. In the program, each pairwise matching information is stored in SiftMatchFile class, which contains the two file names, the number of matched points, each match feature pair has two index and is associated with a flag specifying whether it is a inlier match point based on homography and fundamental matrix, the homography inlier number, the fundamental matrix inlier number, and the homography and fundamental matrix (zeta::Mat3d). Please refer to ```SiftMatchFile::WriteSiftMatchFile``` and ```SiftMatchFile::ReadMatchFile``` for detailed implementations.
## 3. To-do List ##
* Add more image format supports (.png, etc.)
* Optional sift parameters
