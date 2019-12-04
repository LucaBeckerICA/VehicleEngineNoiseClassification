function [data] = subframes(signal, R, N)
% Function that splits the original raw audio signal into subbands of length
% N with overlap R
signal = signal(find(signal,1,'first'):find(signal,1,'last'));
b = mod(length(signal),(N-R));
if b ~= 0
    b = (N-R) - b;
   signal(end + 1:end + b) = 0; 
end
rows = length(signal)/(N-R) - 1;
data = zeros(rows,N);
for n=1:rows %lambda
    data(n,:) = signal((n-1)*R+1:(n-1)*R+N);
end
end