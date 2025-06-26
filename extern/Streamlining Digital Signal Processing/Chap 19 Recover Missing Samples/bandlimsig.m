function x = bandlimsig(N,e)
% Generate a 2N+1 excerpt of a (pi-e)-bandlimited signal.

f = (pi-e)/pi;               % bandlimit fraction 
M = ceil(f*N);               % 2*M+1=number of full band samples
r = randn(2*M+1,1);          % white standard normal signal
x = zeros(2*N+1,1);          % initialize with zero-vector

for k=(-N):1:N
   sincvec=sinc(k*(pi-e)/pi-((-M):1:M)');
   x(k+N+1)=dot(r,sincvec);
end