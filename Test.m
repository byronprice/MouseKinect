% FindMouseDepth.m
%  Subtract the background from the depth images stored in .mat files
%  and find the mask of the mouse's body.

totalFiles = 4000;

fullDepth = zeros(512,424,totalFiles);

index = 1;
for ii=1:20
    filename = sprintf('mouse45140-%d_20170124.mat',ii);
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

% the actual background image
background2 = median(fullDepth(:,:,1:500),3);
background2(background2 == 0) = 800;

% the background as the distance from the sensor to the floor of the arena
background1 = 800.*ones(512,424);
% centerPoint = [512/2,424/2];
% for ii=1:512
%     for jj=1:424
%         % distance from current pixel to center pixels in units of pixels
%         distFromCenter = sqrt((ii-centerPoint(1)).^2+(jj-centerPoint(2)).^2);
%         % convert to units of mm, ~ 900mm / 512 pixels
%         distFromCenter = distFromCenter*1.7578;
%         asqr = 790.^2;
%         bsqr = distFromCenter;
%         c = sqrt(asqr+bsqr);
%         background1(ii,jj) = c;
%     end
% end
% this appears to be a minimal effect

se = strel('disk',3);
for ii=1:totalFiles
   temp = background2-fullDepth(:,:,ii); 
   temp(temp>200) = 0;
   temp(temp<15) = 0;
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

fullDepth = fullDepth(100:450,50:400,:);

forDisplay = uint8((fullDepth./max(max(max(fullDepth)))).*255);

v = VideoWriter('MouseDepthVideo.avi');
v.FrameRate = 60; % really 20 Hz

open(v);
for ii=1:totalFiles
    writeVideo(v,forDisplay(:,:,ii)');
end

close(v);
