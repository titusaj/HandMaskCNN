
function mask = RGBthreshold(rgb_img, thresholds)
%
% INPUTS:
%   rgb_img - image in rgb format
%   thresholds - rgb thresholds, a 6-element vector
%
% OUTPUTS:
%   mask - logical BW image masking out regions that fall within the
%       threshold ranges

r = squeeze(rgb_img(:,:,1));
g = squeeze(rgb_img(:,:,2));
b = squeeze(rgb_img(:,:,3));


 r_mask = (r >= (thresholds(1)-thresholds(2)) & r <= (thresholds(1)+thresholds(2)));
 g_mask = (g >= thresholds(3) & g <= thresholds(4));
 b_mask = (b >= thresholds(5) & b <= thresholds(6));

mask =  r_mask & (~g_mask);