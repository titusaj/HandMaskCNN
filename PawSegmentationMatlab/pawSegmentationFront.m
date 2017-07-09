%Titus John
%Leventhal Lab, University of Michigan
%July 8, 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This assumes the 
% Input
% single video frame

% Output
% paw mask 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [pawMaskLargestBlob,oneBlobCheck]= pawSegmentationFront(rgbImage)

    targetMean = [0.5,0.2,0.5];
    targetSigma = [0.2,0.2,0.2];

    %run a decorr stre
    decorrImage = decorrstretch(rgbImage);

    %pass through an rgb threshold to get mask
    pawRGBrange = [0, 0 ,255, 259, 0,0]; %These setting will be diffrent for the front vs the behind the plexiglass
    pawGreenMask = RGBthreshold(decorrImage,pawRGBrange);

    %Extract the largest green blob from this image 
    [pawMaskLargestBlob,oneBlobCheck] = ExtractNLargestBlobs(pawGreenMask, 1);

    
    

end