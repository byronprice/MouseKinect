//
//  BinaryConversion.cpp
//  
//
//  Created by Byron Price on 2017/01/10.
//
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>

int main(int argc, char *argv[]) {
    int firstFrame = strtol(argv[1],NULL,0);
    int maxFrames = strtol(argv[2],NULL,0);
    
    if (maxFrames==0 || firstFrame == 0) {
        return 0;
    }
    
    for (int ii=firstFrame; ii<maxFrames+1; ii++){
        std::string frameNum = std::to_string(ii);
        std::string filename = "DepthData_" + frameNum + ".bin";
        
        const char *newfilename = filename.c_str();
        FILE *pFile;
        pFile = fopen(newfilename,"rb");
        if (pFile == NULL) {
            fputs (newfilename,stderr);
            fclose(pFile);
        }
        else {
            long lSize;
            unsigned char * buffer;
            size_t result;
            
            fseek(pFile,0,SEEK_END);
            lSize = ftell(pFile);
            
            rewind(pFile);
            buffer = (unsigned char*) malloc (sizeof(char)*lSize);
            result = fread(buffer,1,lSize,pFile);
            
            std::string outputfile = "DepthData_" + frameNum + ".txt";
            std::ofstream myfile;
            myfile.open(outputfile);
            int size = 512*424;
            for (int jj=0; jj<size; jj++) {
                int lowInd = jj*4;
                unsigned char binaryNum[4];
                for (int kk=0; kk<4; kk++) {
                    binaryNum[kk] = buffer[lowInd+kk];
                }
                float myfloat;
                memcpy(&myfloat,&binaryNum,sizeof(myfloat));
                
                myfile << myfloat << "\n";
            }
            myfile.close();
            std::free(buffer);
            fclose(pFile);
        }

        
        std::string rgbfilename = "RGBData_" + frameNum + ".bin";
        const char *rgb_filename = rgbfilename.c_str();
        FILE *rgbFile;
        rgbFile = fopen(rgb_filename,"rb");
        if (rgbFile == NULL) {
            fputs (rgb_filename,stderr);
            fclose(rgbFile);
        }
        else {
            long lSize;
            unsigned char * buffer;
            size_t result;
            
            fseek(rgbFile,0,SEEK_END);
            lSize = ftell(rgbFile);
            
            rewind(rgbFile);
            buffer = (unsigned char*) malloc (sizeof(char)*lSize);
            result = fread(buffer,1,lSize,rgbFile);
            
            std::string outputfile = "RGBData_" + frameNum + ".txt";
            std::ofstream myfile;
            myfile.open(outputfile);
            int size = 1920*1080;
            for (int jj=0; jj<size; jj++) {
                int lowInd = jj*4;
                unsigned char RGB[3];
                float floatRGB[3];
                for (int kk=0; kk<3; kk++) {
                    RGB[kk] = buffer[lowInd+kk];
                    floatRGB[kk] = RGB[kk];
                }
                float gray;
                gray = (floatRGB[0]+floatRGB[1]+floatRGB[2])/3;
                
                myfile << gray << "\n";
            }
            myfile.close();
            std::free(buffer);
            fclose(rgbFile);
        }


    }
    return 0;
}
