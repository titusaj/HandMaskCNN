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
function SingleVideoSegmentation(video)
    
    %video =  VideoReader(videoFileName);%Read the video into the filspace
    numFrames= video.Duration* video.FrameRate; %calculate the tolat number of frames in the video
    
    for i = 1:128%numFrames %loop through the frams of the video to extract the
         rgbImage = read(video, i);
        [pawMaskLargestBlob,oneBlobCheck]= pawSegmentationFront(rgbImage);
        
         i
%         figure(1)
%         imshow(rgbImage)
%         
%          figure(2)
%         imshow(pawMaskLargestBlob)
        
        %Convert the images to uint8
        rgbImage = im2uint8(decorrstretch(rgbImage));
        pawMaskLargestBlob = im2uint8(pawMaskLargestBlob);
       % pawMaskLargestBlob = repmat(pawMaskLargestBlob,[1,1,3]);%Using this to create image with 3 channels for cnn
        
        %filename for rgb image
         rgbFilename = strcat(num2str(i),'.tif');
         rgbImage = rgb2gray(rgbImage);
         rgbImage= imresize(rgbImage, [512, 512]);
         imwrite(rgbImage,rgbFilename); 
        
       
        
%       %filename for mask of paw
%             maskFilename = strcat('1_',num2str(i),'_mask.tif');
%             pawMaskLargestBlob = imresize(pawMaskLargestBlob, [512, 512]);
%             imwrite(pawMaskLargestBlob,maskFilename); 
%         
        
        
    end

end
