% FindMouseDepth.m
%  Subtract the background from the depth images stored in .mat files
%  and find the mask of the mouse's body.

totalFiles = 2000;

fullDepth = zeros(512,424,totalFiles);

index = 1;
for ii=1:20
    filename = sprintf('mouse45180-%d_20170112.mat',ii);
    load(filename,'depthVideo');
    
    fullDepth(:,:,index:index+99) = depthVideo;
    index = index+100;
end

% figure();
% for ii=1:totalFiles
%     imagesc(fullDepth(:,:,ii)');caxis([580 680]);colormap('bone');
%     pause(1/20);
% end

% to make a .avi movie

background2 = median(fullDepth(:,:,1:500),3);
background2(background2 == 0) = 680;

background1 = 680*ones(512,424);

se = strel('disk',3);
for ii=1:totalFiles
   temp = background2-fullDepth(:,:,ii); 
   temp(temp>200) = 0;
   temp(temp<10) = 0;
   temp = imopen(temp,se);
   binaryIm = temp>0;
%    try
%        cc = bwconncomp(binaryIm);
%        numPixels = cellfun(@numel,cc.PixelIdxList);
%        [~,index] = max(numPixels);
%        binaryIm = double(binaryIm);
%        binaryIm(cc.PixelIdxList{index}) = -1;
%        temp(binaryIm~=-1) = 0;
%    catch
%        mask = bwareaopen(binaryIm,200);
%        temp(~mask) = 0;
%    end

   mask = bwareaopen(binaryIm,100);
   temp = background1-fullDepth(:,:,ii);
   temp(~mask) = 0;
   fullDepth(:,:,ii) = medfilt2(temp);
end

fullDepth = fullDepth(115:460,20:370,:);

forDisplay = uint8((fullDepth./max(max(max(fullDepth)))).*255);

v = VideoWriter('MouseDepthVideo.avi');
v.FrameRate = 20;

open(v);
for ii=1:totalFiles
    writeVideo(v,forDisplay(:,:,ii)');
end

close(v);
