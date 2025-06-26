function w = knab(K, beta, d)
% K tap Knab-window with parameter beta and time shift d. For details see:
% Knab, J: "An Alternate Kaiser Window",
% IEEE Trans. on ASSP, Vol. ASSP-27, No. 5, Oct. 1979

if nargin == 2
    d = 0;
end

nrange = ((-(K-1)/2):1:((K-1)/2))';
w = sinh(beta*sqrt(1-(2*(nrange+d)/(K-1)).^2))./(sinh(beta)*sqrt(1-(2*(nrange+d)/(K-1)).^2));

if nrange(1)+d == -(K-1)/2
    w(1) = beta/sinh(beta);
end

if nrange(end)+d == (K-1)/2
    w(end)= beta/sinh(beta);
end