% ReadTxtFilesKinect.m
%  read in the .bin and .txt files from Protonect.cpp and 
%   BinaryConversion.cpp, then convert those to .mat format

depthFiles = dir('DepthData_*.bin');
numDepthFiles = length(depthFiles);

rgbFiles = dir('RGBData_*.bin');
numrgbFiles = length(rgbFiles);

fileIters = 100;
numIter = round(numDepthFiles/fileIters);

allDepthNums = zeros(numDepthFiles,1);
for ii=1:numDepthFiles
    lowIndex = regexp(depthFiles(ii).name,'_');
    highIndex = regexp(depthFiles(ii).name,'.bin');
    allDepthNums(ii) = str2double(depthFiles(ii).name(lowIndex+1:highIndex-1));
end
allDepthNums = sort(allDepthNums);

allrgbNums = zeros(numDepthFiles,1);
for ii=1:numDepthFiles
    lowIndex = regexp(rgbFiles(ii).name,'_');
    highIndex = regexp(rgbFiles(ii).name,'.bin');
    allrgbNums(ii) = str2double(rgbFiles(ii).name(lowIndex+1:highIndex-1));
end
allrgbNums = sort(allrgbNums);

depth_width = 512;
depth_height = 424;

rgb_width = 1920/3;
rgb_height = 1080/3;

Date = datetime('today','Format','yyyy-MM-dd');
Date = char(Date); Date = strrep(Date,'-','');Date=str2double(Date);

globalCount = 1;
for jj=1:numIter
    depthVideo = zeros(depth_width,depth_height,fileIters);
    rgbVideo = zeros(rgb_width,rgb_height,fileIters);
    
    rgbFrames = zeros(fileIters,1);
    depthFrames = zeros(fileIters,1);

    for ii=1:fileIters
        depthFrames(ii) = allDepthNums(globalCount);
        filename = sprintf('DepthData_%d.txt',allDepthNums(globalCount));
        fileID = fopen(filename,'r');
        formatSpec = '%f';
        Z = fscanf(fileID,formatSpec);
        fclose(fileID);
        Z = reshape(Z,[depth_width,depth_height]);
        depthVideo(:,:,ii) = Z;
        
        rgbFrames(ii) = allrgbNums(globalCount);
        filename = sprintf('RGBData_%d.txt',allrgbNums(globalCount));
        fileID = fopen(filename,'r');
        formatSpec = '%f';
        Z = fscanf(fileID,formatSpec);
        fclose(fileID);

        Z = reshape(Z,[rgb_width*3,rgb_height*3]);
        rgbVideo(:,:,ii) = Z(1:3:end,1:3:end);
        
        globalCount = globalCount+1;
    end
    
    figure();
    for ii=1:fileIters
        subplot(1,2,1);
        imagesc(depthVideo(:,:,ii)');caxis([50 80]);colormap('hsv');
        subplot(1,2,2);
        imagesc(rgbVideo(:,:,ii)');colormap('bone');
        pause(1/20);
    end
    
    
    fileName = sprintf('mouse45180-%d_%d.mat',jj,Date);
    save(fileName,'depthVideo','rgbVideo','rgbFrames','depthFrames');
end

condition = input('Delete all raw data files? (y/n): ','s');

if condition == 'y'
    delete RGBData*
    delete DepthData*
end
