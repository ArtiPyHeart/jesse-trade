
% Matlab code for article "Accurate, Guaranteed-Stable, 
%  Sliding DFT", by Krzysztof Duda

clear, clc
%% Define DFT parameters
M = 30; 			%DFT length
k = 3;			%DFT bin to be computed
r = 0.99999;		%for r<1 the pole is moved inside the unit circle

W = exp(j*2*pi/M);	%twiddle factor

%% Create a test input signal
Npts = 206;       % Number of test signal samples
x = sin(2*pi*k*(0:Npts/2-1)/M).*(0:Npts/2 -1)/Npts;	% Random test signal
x = [x, fliplr(x)];	% Random test signal

%x = rand(1,Npts);	% Random test signal

% Plot input sequence
figure(1)
plot(x, '-bo', 'markersize', 4)
title('Input sequence')
xlabel('Time')
ylabel('Amplitude')
grid on, zoom on

%% Initial values
xmr = 1;
X0R = 0;
X0M = 0;
X0D = 0;
xn_M= 0;
fifo= zeros(1,M);

for n=1:length(x);
%% FIFO buffer
	fifo= [fifo(2:M) x(n)];
%% SDFT for r=1 or rSDFT for r<1
	X0R = r*W^(k)*(X0R+x(n)-r^M*xn_M); %(11)
SDFT_Out_Mag(n) = abs(X0R);          % Output magnitude
SDFT_Out_Phase(n) = atan2(imag(X0R), real(X0R))*180/pi; % Output phase

%% mSDFT
	X1M = X0M+xmr*( x(n)-xn_M ); % Fig.1(d)
	X0M = X1M;
	if mod(n,M)
		xmr=xmr*W^(-k);		%modulating sequence
		X1M = X1M*conj(xmr);	%phase correction
	else
		xmr=1;
	end
mSDFT_Out_Mag(n) = abs(X1M);          % Output magnitude
mSDFT_Out_Phase(n) = atan2(imag(X1M), real(X1M))*180/pi; % Output phase

%% S. Douglas and J. Soh algorithm
	if mod(n,M)==0
		X1D = r*W^k*(X0D-xn_M)+ W^k*x(n); %(15)
	else
		X1D = W^k*(X0D-r*xn_M+x(n)); %(15)
 	end
	X0D = X1D;
D_and_S_Out_Mag(n) = abs(X1D);          % Output magnitude
D_and_S_Out_Phase(n) = atan2(imag(X1D), real(X1D))*180/pi; % Output phase

%% new xn_M
 	xn_M = fifo(1);
end

figure(2)
subplot(2,1,1)
plot(SDFT_Out_Mag, '-bo', 'markersize', 8)
hold on
plot(mSDFT_Out_Mag, '-kd', 'markersize', 6)
plot(D_and_S_Out_Mag, '-r*', 'markersize', 4)
hold off
legend('SDFT', 'mSDFT', 'D & S')
title(['DFT Results:  (M = ',num2str(M),',   k = ',num2str(k),')'])
xlabel('Time')
ylabel('Mag (Lin.)')
grid on, zoom on

subplot(2,1,2)
plot(SDFT_Out_Phase, '-bo', 'markersize', 8)
hold on
plot(mSDFT_Out_Phase, '-kd', 'markersize', 6)
plot(D_and_S_Out_Phase, '-r*', 'markersize', 4)
hold off
grid on, zoom on
xlabel('Time')
ylabel('Phase (Deg.)')

