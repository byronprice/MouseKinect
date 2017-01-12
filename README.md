# MouseKinect
Code to track a mouse and quantify behavior using an XBOX Kinect.

You'll need to download libfreenect2 from https://github.com/OpenKinect/libfreenect2 . Once this is downloaded to your computer, you can build the version of Protonect.cpp in this repository. It is designed to save the depth and RGB data streams coming off of the Kinect. Next, build BinaryConversion.cpp to convert the files from .bin to .txt . Finally, use ReadTxtFilesKinect to convert the .txt files to downsampled .mat files. (Future version will likely go directly from .bin to .mat . 
