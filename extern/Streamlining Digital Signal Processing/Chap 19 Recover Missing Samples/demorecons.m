% Demo-script showing functionality of filter bank reconstruction method.
% Requires the files "knab.m", "interpfbank.m", "bandlimsig.m" and
% "recons.m" in a Matlab-visible directory.
%
% Try changing the "tpind" variable for different reconstruction setups.
% The guard-band fraction variable "f" should be slightly larger than 1.
% Setting "f" to a value smaller than 1.0 simulates the effect of not
% satisfying the bandwidth limitation required for perfect reconstruction.
% The reconstruction filter order "Ntaps" can also be adjusted.
%
% Also requires the "pwelch" function from the Signal Processing Toolbox.
%
% Description of files:
% knab.m          - Generate shifted window for windowed-sinc filter design.
% interpfbank.m   - Generates the filterbanks' coefficients.
% recons.m        - Performs filter bank reconstruction of a signal.
% bandlimsig.m    - Generate bandlimited signal for demo purposes.
%
% Author: Andor Bariska, 11/JUNE/2007

clear, clc

% Start by defining the available and missing samples. This is done in
% a 0-1 indicator vector, which has a "1" to indicate an available sample
% and a "0" to indicate a missing sample. Only one recurrence period is
% needed:
tpind = [1 1 0 1 0];           % example with A=3 available and M=2 missing samples.
A = sum(tpind);                % A = number of availble data points
T = length(tpind);             % T = recurrence period

% Generate a appropriately bandlimited test signal with no missing samples:
f = 1.5;                      % Define how much guard-band is in the signal.
DataPoints = 2001;             % Number of data points for the simulation.
x = bandlimsig((DataPoints-1)/2,f*(T-A)/T*pi);

% display the demo data (requires Signal Processing Toolbox):
figure(1)
subplot(2,1,1)
plot(x)
title('Random Bandlimited Signal')
axis([0 DataPoints-1 -3 3])
subplot(2,1,2)
pwelch(x)

% Calculate filter bank coeffs based on recurrence period indicator:
Ntaps = 17;   % MUST BE ODD - number of filter taps
K = (Ntaps-1)/2;
h=interpfbank(tpind,K); % h contains one filter per column

% Calculate reconstruction y:
y=recons(x,tpind,h);

% Display error of reconstructed signal:
ReconstructionRange = T*Ntaps:DataPoints-T*Ntaps;   % range where filter output is valid

figure(2)
subplot(2,1,1);
plot(x(ReconstructionRange))
hold on
plot(y(ReconstructionRange),'r')
hold off
title('Original and Reconstructed Signals')
legend('Original Data','Reconstructed Data')
subplot(2,1,2);
plot( abs(x(ReconstructionRange) - y(ReconstructionRange)) );
title('Absolute Steady-State Reconstruction Error')


