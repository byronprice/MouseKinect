% FindMouseDepth.m
%  Subtract the background from the depth images stored in .mat files
%  and find the mask of the mouse's body.

v = VideoWriter('MouseDepthVideo.avi');
v.FrameRate = 20; % really 20 Hz
open(v);

index = 1;
for ii=1:40
    filename = sprintf('mouse45140-%d_20170124.mat',ii);
    display(filename);
    load(filename,'depthVideo');
    totalFiles = size(depthVideo,3);
    
    
    floor = 800;
    % the actual background image
    background2 = median(depthVideo,3);
    background2(background2 == 0) = floor;
    
    % the background as the distance from the sensor to the floor of the arena
    background1 = floor.*ones(512,424);
    
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
    for jj=1:totalFiles
        temp = background2-depthVideo(:,:,jj);
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
        temp = background1-depthVideo(:,:,jj);
        temp(~mask) = 0;
        temp = medfilt2(temp);
        
        temp = temp(100:450,50:400);
        
        forDisplay = uint8((temp./max(max(temp))).*255);
        
        
        writeVideo(v,forDisplay(:,:)');
    end
end
% figure();
% for ii=1:totalFiles
%     imagesc(fullDepth(:,:,ii)');caxis([580 680]);colormap('bone');
%     pause(1/20);
% end


close(v);
