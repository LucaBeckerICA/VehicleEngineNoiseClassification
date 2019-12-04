function [data] = SCMS(low_level_features,S,K)
%Computes the short-time modulation spectrum
%Input parameters are the Shift amount (scalar) and the underlying low-level features (PCEN/MFCCs)
%(matrix)
s = S;
k = K;

for n=1:size(low_level_features, 1)
    temp(n,:,:) = subframes(low_level_features(n,:),s,k);
end
%data = zeros(s(1), k, s(2));
for n=1:size(temp,1) %q
    for m=1:size(temp,2)
        data(n,m,:) = fft(temp(n,m,:));
    end
end


