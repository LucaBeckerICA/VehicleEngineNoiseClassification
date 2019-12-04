function [data] = meanSCMS(scms)
% Function that averages over the temporal index of the Modulation Spectrum (cf. Becker et al.: Eq. 3)
s = size(scms);
data = zeros(s(1), s(3));
for n=1:s(1)
    for m=1:s(3)
        data(n,m) = mean(abs(scms(n,:,m)));
    end
end