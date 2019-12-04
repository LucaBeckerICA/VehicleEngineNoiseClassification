function [data] = modMFCCMaps(low_level_feature_cell, name)

% Function that computes the Modulation Spectra for each element in low_level_feature_cell. The shape of the low_level_feature_cell
% should be {1,n}, whereas each element n should be a matrix with dimensions [spectral_bins, temporal_bins]
general_shape = size(mfcc_cell);
shape = size(mfcc_cell{1,1});
% K should be even, can be an arbitrary value
K = shape(2);
S = K/2;
data = cell(general_shape);
    for n=1:general_shape(2)
        temp = SCMS(mfcc_cell{1,n}',S,K);
        data{1,n} = meanSCMS(temp);
    end
save(name,'data');
end