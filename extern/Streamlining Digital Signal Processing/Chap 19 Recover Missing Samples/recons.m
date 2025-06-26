function y = recons(x, tpind, h)
% Reconstruct the periodically incompletely sampled sequence x,
% with periodic sampling determined by the indicator vector tpind,
% using the reconstruction filterbank h.
%
% Example for tpind = [1 1 0 1 0]. It is assumed that x starts with 
% the start of a period.
%
% Replaces the samples with 0 in tpind with the filterbank output.

N = sum(tpind);                 % N=number of available samples.
T = length(tpind);              % T=recurrence period
M = T-N;                        % M=number of missing samples.

tp = find(tpind);               % tp contains the available sample times
tm = find(abs(tpind-1));        % tm contains the missing sample times.

NumberOfPeriods = ceil(length(x)/T);
[r,c] = size(x);
if c > 1
    disp('Transposing x ...');
    x=x';
end 

% zero-pad if length is not a multiple of recurrence period:
if (length(x) - NumberOfPeriods * T) < 0
    x=[x; zeros(NumberOfPeriods * T - length(x),1)];
end

[K,c] = size(h);                % K=length of reconstruction filters (must be odd)
D=(K-1)/2;                      % D=filter bank delay [in periods]

% reconstruction loop over number of recurrence periods:
n=(D+1):(NumberOfPeriods-D);      % indexing range for filtering
y=x;                              % init reconstruction output
% y=zeros(length(x),1);
    
% loop over the number of missing samples per period (=M):
for m=1:M
    
    FilterOutput = zeros(NumberOfPeriods,1);    % init filter output
    
    % loop over the number of available samples per period (=N):
    for p=1:N                   % sum over p of x_m[n,p] convolved with h_{t_m,p)
        FilterOutput = conv( x( n*T + tp(p) ), h( :, p+(m-1)*N ) ) + FilterOutput;
    end
    y( (1:(NumberOfPeriods))*T + tm(m) ) = FilterOutput;
end