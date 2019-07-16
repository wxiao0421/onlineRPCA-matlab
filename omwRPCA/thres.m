function [y] = thres(x, mu)
% 
%     y = sgn(x)max(|x| - mu, 0)
%     
%     Parameters
%     ----------
%     x: numpy array
%     mu: thresholding parameter
%     
%     Returns:
%     ----------
%     y: numpy array

%     y = max(x - mu, 0);
%     y = y + min(x + mu, 0);
    y = sign(x).*max(abs(x)-mu,0);
end