function h = interpfbank(tpind, K)
% Windowed sinc fractional delay interpolation filter bank. tpind is 
% the indicator vector for the available sample indeces, and 2*K+1
% the FIR filter order.
%
% Example for tpind = [1 1 0 1 0] (same as in the paper).
%
% Author: Andor Bariska, 11/JUNE/2007

A = sum(tpind);                 % A=number of available samples.
T = length(tpind);              % T=recurrence period
M = T-A;                        % M=number of missing samples.

tA = find(tpind)-1;             % tA contains the available sample times
tM = find(abs(tpind-1))-1;      % tM contains the missing sample times.

% Initialization
k = (-K:1:K)';             % range for the index variable n
cs= (-1).^( k*(A-1) );     % cs = cos(pi n (A-1));
h = zeros(2*K+1,M*A);

% compute gain factors Ga,m:
G=ones(A,M);
for m=1:M
for a=1:A
    for q=1:A
        if q==a
            continue;
        end
        G(a,m)=G(a,m)*sin(pi*(tM(m)-tA(q))/T)/sin(pi*(tA(a)-tA(q))/T);
    end
end
end

% generate filter coefficients for reconstruction filter bank:
for m=1:M
    for a=1:A
        Delta_am = (tM(m)-tA(a))/T;  % fractional delay value
        h(:,a+(m-1)*A) = G(a,m)*cs.*sinc(Delta_am + k).*knab(2*K+1,15,Delta_am);
    end
end

