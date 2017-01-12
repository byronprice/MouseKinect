% BatchAnalysisMouseCov.m
%  Use with BatchRunMouseDetect.m
%   That file will output a series of files named
%   'mouse40710_8CovMats1.mat' with the mouse ID identified after 'mouse',
%   the number of features in the covariance matrix identified after the
%   underscore and the video number after 'Mats'.

%  This code will use k-means clustering to identify clusters within the
%  set of covariance matrices.

% Created by: Charles Saunders & Byron Price
%  2016/12/14
mouseNum = [40710,40711,40712,40713];

%% COMPILE ALL COVARIANCE MATRICES
numFeatures = 8;
allCovMats = [];
allInds = [];
for mouse = mouseNum
    for video = 1:10
        clearvars -except mouse video mouseNum allCovMats allInds numFeatures
        load(sprintf('mouse%d_%dCovMats%d.mat',mouse,numFeatures,video));
        maxIms = size(upperTriCovVals,2);
        for ii=1:maxIms
            if squeeze(upperTriCovVals(:,ii)) ~= zeros(numFeatures*(numFeatures+1)/2,1)
                allCovMats = [allCovMats;upperTriCovVals(:,ii)'];
                allInds = [allInds;[mouse,video,ii]];
            end
        end
    end
end
numIms = size(allCovMats,1);

% run PCA to look for clusters visually
[coeff,score,latent] = pca(allCovMats);
figure();scatter3(score(:,1),score(:,2),score(:,3));
title('Principal Component Deconstruction of Covariance Matrices');
xlabel('PC1');ylabel('PC2');zlabel('PC3');
% index = find(latent<0.01,1,'first');
% newData = score(:,1:index)*coeff(:,1:index)';

% calculate all-possible distances between matrices
distMat = zeros(numIms,numIms);
for ii=1:numIms
    for jj=1:numIms
        for kk=1:(numFeatures*(numFeatures+1)/2)
            distMat(ii,jj) = distMat(ii,jj)+(allCovMats(ii,kk)-allCovMats(jj,kk)).^2;
        end
    end
end
figure();imagesc(distMat);colorbar;
title('All-Possible Euclidean Distances between Covariance Matrices');
xlabel('Image Number');ylabel('Image Number');

%% k-means clustering across a number of different cluster sizes, k
maxClusters = 50;
bigSum = zeros(maxClusters,1);
% meanSilh = zeros(maxClusters,1);
for ii=1:maxClusters
    [idx,C,sumd] = kmeans(allCovMats,ii,'Distance','sqeuclidean','Replicates',20);
    bigSum(ii) = sum(sumd);
%     figure;
%     [silh,h] = silhouette(allCovMats,idx,'sqeuclidean');
%     h = gca;
%     h.Children.EdgeColor = [.8 .8 1];
%     xlabel 'Silhouette Value';
%     ylabel 'Cluster';
%     title(sprintf('%d Clusters',ii));
%     meanSilh(ii) = mean(silh);
end

% figure();plot(1:maxClusters,meanSilh);title('Mean Silhouette Values');
% xlabel('Number of Clusters (k)');ylabel('Mean');
figure();plot(1:maxClusters,bigSum,'LineWidth',2);title('Total Distance from Every Point to Its Centroid');
xlabel('Number of Clusters (k)');ylabel('Sum');

% best to halt code and look for appropriate number of clusters, 
%  then repeat the clustering algorithm for a given number of clusters

%% DISPLAY CLUSTER EXEMPLARS
indeces = unique(idx);
percentages = zeros(length(indeces),1);
clusterGroups = cell(length(indeces),1);
for ii=1:length(indeces)
    Inds = find(idx==ii);
    percentages(ii) = length(Inds)./numIms;
    clusterGroups{ii} = Inds;
    
    randNums = random('Discrete Uniform',length(Inds),[10,1]);
    figure();
    for jj=1:10
        tempInd = Inds(randNums(jj));
        load(sprintf('mouse%d_%dCovMats%d.mat',allInds(tempInd,1),numFeatures,allInds(tempInd,2)));
        subplot(5,2,jj);imagesc(finalImages{allInds(tempInd,3)});colormap('bone');
    end
end

% display cluster probabilities
figure();bar(percentages);title('Probability of Cluster Exemplar Appearing in Dataset');
xlabel('Cluseter Number');ylabel('Probability');