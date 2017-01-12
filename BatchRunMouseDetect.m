% BatchRunMouseDetect.m
%  Find a mouse within an image, detect it's head position and orientation, 
%   then calculate a region descriptor covariance matrix to classify its
%   pose.
%  
%   Data is saved as 'mouse40710_webcam1.mat' for different mice, with the 
%    identifier at the end ranging from 1 to 10. Each .mat file has a set
%    of 100 images in a 480x640x3x100 vector called colour_data.
%
%   The algorithm is highly dependent on the color of the mouse being
%    distinct from the background. In our case, Black6 mice are viewed from
%    above with a white background.

% Created by: Charles Saunders & Byron Price
%  2016/12/14

mouseNum = [40710,40711,40712,40713];

numFeatures = 8;
for mouse = mouseNum
    for video = 1:10
        clearvars -except mouse video mouseNum numFeatures
        % load file
        load(sprintf('mouse%d_webcam%d.mat',mouse,video));
        
        background = median(colour_data,4);
        %Subtract background ... in some cases the mouse doesn't move much
        % during the 100 images and so shows up in the background image.
        % Best practice is to check the background image to make sure it is
        % mouse free.
        background_subtracted = zeros(size(colour_data));
        for i=1:size(colour_data,4)
            background_subtracted(:,:,:,i) = abs(colour_data(:,:,:,i) - background);
        end
        
        % initialize variables that will be stored
        maxIms = size(colour_data,4);
        covMats = zeros(numFeatures,numFeatures,maxIms);
        logCovMats = zeros(numFeatures,numFeatures,maxIms);
        upperTriCovVals = zeros(numFeatures*(numFeatures+1)/2,maxIms);
        finalImages = cell(maxIms,1);
        for imNum = 1:maxIms
            %choose 1 image at a time
            image_noback = double(background_subtracted(:,:,:,imNum));
            
            % calculate luminance from color image
            colorDist = mean(image_noback,3);
            
            %% MOUSE DETECTION
            
            % threshold the luminance values to find the mouse and cut out
            %  both noise and shadows/reflections
            temp = colorDist(:);temp = temp(temp>15);
            try
                GMModel = fitgmdist(temp,3);
                mu = GMModel.mu;Sigma = squeeze(GMModel.Sigma);
                [~,minInd] = min(mu);
                threshold = mu(minInd)+4*sqrt(Sigma(minInd));
                if threshold < 40 || threshold > 85
                    threshold = 40;
                end
            catch
                threshold = 40;
            end
            colorDist(colorDist<threshold) = 0; 
            
            % find the largest connected component remaining after
            %  thresholding, i.e. the mice
            tempBinary = double(colorDist>0);
            try
                cc = bwconncomp(tempBinary);
                numPixels = cellfun(@numel,cc.PixelIdxList);
                [~,index] = max(numPixels);
                tempBinary(cc.PixelIdxList{index}) = -1;
                colorDist(tempBinary~=-1) = 0;
            catch
                mask = bwareaopen(colorDist,100);
                colorDist(~mask) = 0;
            end
            
            % eliminate those images in which the mouse is close to the
            %  edge of the arena (in this particular dataset, when the mouse
            %  is to the edge it's body is often obscured)
            EdgeMask = zeros(size(colorDist));
            EdgeMask(1:2,:) = 1;EdgeMask(end-1:end,:) = 1;
            EdgeMask(:,1:2) = 1;EdgeMask(:,end-1:end) = 1;

            binaryMask = colorDist>0;
            temp = binaryMask.*EdgeMask;
            numPixels = sum(sum(temp));
            
            if numPixels > 40
                continue;
            else
                %% TAIL DETECTION
                
                % convolution to find tail
                color_binary = double(colorDist>0);
                window = 15;
                h = ones(window,window);
                convIm = filter2(h,color_binary,'same');
                convIm(color_binary==0) = 0;
                result = convIm;
                
                % Gaussian mixture model for adaptive threshold to
                %  segregate the body from the tail
                temp = result(:);temp = temp(temp~=0);
                try
                    GMModel = fitgmdist(temp,2);
                    mu = GMModel.mu;Sigma = squeeze(GMModel.Sigma);
                    [~,minInd] = min(mu);
                    threshold = mu(minInd)+3*sqrt(Sigma(minInd));
                    
                    if threshold < 40 || threshold > window^2/2
                        threshold = 80;
                    end
                    % display GMModel
                    %     Y = pdf(GMModel,(0:max(temp))');
                    %     figure();plot(0:max(temp),Y);title('Mixture Model Fit for Tail');
                catch
                    threshold = 80;
                end
                
                tail_mask = result; % Get the mask for the tail
                tail_mask(result>threshold) = 0; %Thresholding
                temp = double(tail_mask>0);
                
                % the tail is the largest connected component remaining
                %  after thresholding
                try
                    cc = bwconncomp(temp);
                    numPixels = cellfun(@numel,cc.PixelIdxList);
                    [~,index] = max(numPixels);
                    temp(cc.PixelIdxList{index}) = -1;
                    tail_mask(temp~=-1) = 0;
                catch
                    [~,ind] = min(result);
                    tail_mask(ind) = 1;
                end
                
                % get the location of the base of the tail
                [~, idx] = max(tail_mask(:));
                [tail_y, tail_x] = ind2sub(size(tail_mask),idx);
                
                %Remove tail from the original image of the whole mouse
                body_image = colorDist;
                body_image(tail_mask>0) = 0;
                
                %% DETERMINE POINTING DIRECTION AND HEAD LOCATION
                
                % Create Point cloud
                [r,c] = find(body_image~=0);
                cloud = [c,r];
                cloud_covariance = cov(cloud);
                
                [eigenvectors,eigenvalues] = eigs(cloud_covariance); %PCA

                % Determining pointing direction
                com_body = [mean(cloud(:,1)),mean(cloud(:,2))];
                
                %Make sure vector is pointing towards head 
                % (from base of tail to center of body mass)
                dotproduct = dot((com_body-[tail_x, tail_y])/norm(com_body-...
                    [tail_x, tail_y]),eigenvectors(:,2)/norm(eigenvectors(:,2)));
                if (dotproduct<0)
                    eigenvectors(:,2) = -eigenvectors(:,2);
                end

                % %Plot the direction vectors
                % figure();
                % scatter(cloud(:,1),cloud(:,2))
                % hold on
                % s1 = 2*sqrt(eigenvalues(1,1));
                % s2 = 2*sqrt(eigenvalues(2,2));
                % line(mean(cloud(:,1))+[eigenvectors(1,1)*-s1,eigenvectors(1,1)*s1],mean(cloud(:,2))+[eigenvectors(2,1)*-s1,eigenvectors(2,1)*s1],'Color','r','LineWidth',2)
                % line(mean(cloud(:,1))+[eigenvectors(1,2)*-s2,eigenvectors(1,2)*s2],mean(cloud(:,2))+[eigenvectors(2,2)*-s2,eigenvectors(2,2)*s2],'Color','g','LineWidth',2)
                % axis equal
                % set(gca,'Ydir','reverse')
                %
                % hold on
                % scatter(com_body(1),com_body(2),'xr','LineWidth',3)
                %
                % %Head point
                % head = [mean(cloud(:,1))+eigenvectors(1,2)*s2,mean(cloud(:,2))+eigenvectors(2,2)*s2];
                % scatter(head(1),head(2),'or','LineWidth',28)
                % hold off;

                %% IMAGE ROTATION
                [r,c] = find(body_image);
                image_to_rotate = padarray(body_image(min(r(:)):max(r(:)),min(c(:)):max(c(:))),[100,100]);
                
                % get body angle from eigenvectors
                bodyAngle = atan(eigenvectors(2,2)/eigenvectors(1,2));
                
                % convert bodyAngle from 2 quadrant to 4 quadrant
                %  representation
                if eigenvectors(1,2) >= 0 && eigenvectors(2,2) > 0
                    bodyAngle = bodyAngle;
                elseif eigenvectors(1,2) >= 0 && eigenvectors(2,2) < 0
                    bodyAngle = bodyAngle;
                elseif eigenvectors(1,2) < 0 && eigenvectors(2,2) >= 0
                    bodyAngle = bodyAngle+pi;
                elseif eigenvectors(1,2) < 0 && eigenvectors(2,2) <= 0
                    bodyAngle = pi+bodyAngle;
                end
                
                % rotate image
                rotated_image = imrotate(image_to_rotate,radtodeg(bodyAngle));
                [r,c] = find(rotated_image~=0);
                final_image=rotated_image(min(r(:)):max(r(:)),min(c(:)):max(c(:)));

                %% COVARIANCE DESCRIPTOR
                finalImages{imNum} = final_image;
                feature_des_south= zeros(size(final_image));
                
                %SOUTH
                for ix=1:size(final_image,2) %Loop through each column
                    
                    point1 = 1; %Find first point of mouse
                    cpoint = final_image(point1,ix);
                    while(cpoint == 0 && point1<size(final_image,1))
                        point1 = point1 + 1;
                        cpoint = final_image(point1,ix);
                    end
                    
                    for iy=point1:size(final_image,1) %Find distance from current pixel to second edge
                        dist = 0;
                        cpoint = final_image(iy+dist,ix);
                        while(cpoint~=0 && iy+dist<size(final_image,1))
                            dist = dist + 1;
                            cpoint = final_image(iy+dist,ix);
                        end
                        feature_des_south(iy,ix) = dist;
                    end
                end
                
                %NORTH
                feature_des_north= zeros(size(final_image));
                
                for ix=1:size(final_image,2) %Loop through each column
                    
                    point1 = 1; %Find first point of mouse
                    cpoint = final_image(point1,ix);
                    while(cpoint == 0 && point1<size(final_image,1))
                        point1 = point1 + 1;
                        cpoint = final_image(point1,ix);
                    end
                    
                    for iy=point1:size(final_image,1) %Find distance from current pixel to second edge
                        dist = 0;
                        cpoint = final_image(iy-dist,ix);
                        while(cpoint~=0 && iy-dist>1)
                            dist = dist + 1;
                            cpoint = final_image(iy-dist,ix);
                        end
                        feature_des_north(iy,ix) = dist;
                    end
                end
                
                %EAST
                feature_des_east= zeros(size(final_image));
                
                for iy=1:size(final_image,1) %Loop through each row
                    
                    point1 = 1; %Find first point of mouse
                    cpoint = final_image(iy,point1);
                    while(cpoint == 0 && point1<size(final_image,2))
                        point1 = point1 + 1;
                        cpoint = final_image(iy,point1);
                    end
                    
                    for ix=point1:size(final_image,2) %Find distance from current pixel to second edge
                        dist = 0;
                        cpoint = final_image(iy,ix+dist);
                        while(cpoint~=0 && ix+dist<size(final_image,2))
                            dist = dist + 1;
                            cpoint = final_image(iy,ix+dist);
                        end
                        feature_des_east(iy,ix) = dist;
                    end
                end
                
                %WEST
                feature_des_west= zeros(size(final_image));
                
                for iy=1:size(final_image,1) %Loop through each row
                    
                    point1 = 1; %Find first point of mouse
                    cpoint = final_image(iy,point1);
                    while(cpoint == 0 && point1<size(final_image,2))
                        point1 = point1 + 1;
                        cpoint = final_image(iy,point1);
                    end
                    
                    for ix=point1:size(final_image,2) %Find distance from current pixel to second edge
                        dist = 0;
                        cpoint = final_image(iy,ix+dist);
                        while(cpoint~=0 && ix-dist>1)
                            dist = dist + 1;
                            cpoint = final_image(iy,ix-dist);
                        end
                        feature_des_west(iy,ix) = dist;
                    end
                end
                
                %X pos
                feature_des_x = zeros(size(final_image));
                
                for ix = 1:size(final_image,2)
                    for iy = 1:size(final_image,1)
                        if final_image(iy,ix) ~= 0
                            feature_des_x(iy,ix) = ix;
                        end
                    end
                end
                
                %Y pos
                feature_des_y = zeros(size(final_image));
                
                for ix = 1:size(final_image,2)
                    for iy = 1:size(final_image,1)
                        if final_image(iy,ix) ~= 0
                            feature_des_y(iy,ix) = iy;
                        end
                    end
                end
                
                % horizontal and vertical edges from first derivative
                h_horz = fspecial('prewitt');h_vert = h_horz';
                feature_des_horzedge = filter2(h_horz,final_image,'same');
                feature_des_vertedge = filter2(h_vert,final_image,'same');
                
                % edges from second derivative
                h_laplace = fspecial('laplacian');
                feature_des_laplace = filter2(h_laplace,final_image,'same');
                
                % luminance
                feature_des_luminance = final_image;
                
                % get vector of only the features, with the zeros beyond
                %  the mouse's body removed
                Inds = find(final_image>0);
                feature_des_east = feature_des_east(Inds);
                feature_des_south = feature_des_south(Inds);
                feature_des_north = feature_des_north(Inds);
                feature_des_west = feature_des_west(Inds);
                feature_des_x = feature_des_x(Inds);
                feature_des_y = feature_des_y(Inds);
                feature_des_luminance = feature_des_luminance(Inds);
                feature_des_horzedge = feature_des_horzedge(Inds);
                feature_des_vertedge = feature_des_vertedge(Inds);
                feature_des_laplace = feature_des_laplace(Inds);
                
                %% COVARIANCE CALCULATION
                N = length(feature_des_east(:))-1;
                feature_matrix = [feature_des_east(:)-mean(feature_des_east(:)),...
                    feature_des_west(:)-mean(feature_des_west(:)),...
                    feature_des_north(:)-mean(feature_des_north(:)),...
                    feature_des_south(:)-mean(feature_des_south(:)),...
                    feature_des_x(:)-mean(feature_des_x(:)),...
                    feature_des_y(:)-mean(feature_des_y(:)),...
                    feature_des_luminance(:)-mean(feature_des_luminance(:)),...
                    feature_des_horzedge(:)-mean(feature_des_horzedge(:)),...
                    feature_des_vertedge(:)-mean(feature_des_vertedge(:)),...
                    feature_des_laplace(:)-mean(feature_des_laplace(:))];
                
                covariance_matrix = (feature_matrix'*feature_matrix)./N;
                
                % log covariance conversion
                %  1) singular value decomposition of the covariance matrix
                %    C = U*S*V'
                [U,S,V] = svd(covariance_matrix);
                S_prime = zeros(size(S));
                for ii=1:length(S)
                    if S(ii,ii) < 1e-10
                        S_prime(ii,ii) = log(1e-10);
                    else
                        S_prime(ii,ii) = log(S(ii,ii));
                    end
                end
                
                % 2) conversion back to log(C) = U*log(S)*V'
                log_covariance = U*S_prime*V';
                
                % 3) get only upper triangular values
                temp = ones(numFeatures,numFeatures);
                indeces = find(triu(temp));
                upperTri = log_covariance(indeces);
                logcov_values = upperTri;
                
                covMats(:,:,imNum) = covariance_matrix;
                logCovMats(:,:,imNum) = log_covariance;
                upperTriCovVals(:,imNum) = logcov_values;
            end
        end
        % save the converted images and covariance matrices
        filename = sprintf('mouse%d_%dCovMats%d.mat',mouse,numFeatures,video);
        save(filename,'covMats','logCovMats','upperTriCovVals','finalImages');
    end
end
%  to view rotated images as a kind-of video
% indeces = [];
% for ii=1:100
% if squeeze(covMats(:,:,ii)) ~= zeros(6,6)
% indeces = [indeces,ii];
% end
% end
% figure();
% for ii=indeces
% imagesc(finalImages{ii});pause(0.2);
% end

