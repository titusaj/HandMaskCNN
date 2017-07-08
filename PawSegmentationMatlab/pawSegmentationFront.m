%Titus John
%Leventhal Lab, University of Michigan
%July 8, 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input
% single video frame

% Output
% paw mask 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function []= pawSegmentationFront(rgbImage)

targetMean = [0.5,0.2,0.5];
targetSigma = [0.2,0.2,0.2];

%run a decorr stre
decorrImage = decorrstretch(rgbImage);

%pass through an rgb threshold to get mask
pawRGBrange = [0, 0 ,255, 259, 0,0];
pawMask = RGBthreshold(decorrImage,pawRGBrange)

end