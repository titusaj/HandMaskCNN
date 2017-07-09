%Titus John
%Leventhal Lab, University of Michigan
%July 8, 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input
% Video file for a given view of the mask

% Output
% This takes a single video and output the groundtruth paw segmentation
% This can be done manullay or using computer vision segmentation



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function SingleVideoSegmentation(videoFileName)
    
    video =  VideoReader(videoFileName);%Read the video into the filspace
    numFrames= video.Duration* video.FrameRate; %calculate the tolat number of frames in the video
    
    for i = 1:numFrames %loop through the frams of the video to extract the 
        rgbImage = read(video, i);
        [pawMaskLargestBlob,oneBlobCheck]= pawSegmentationFront(rgbImage);
        
        figure(1)
        imshow(rgbImage)
        
        figure(2)
        imshow(pawMaskLargestBlob)
        
    end

end
