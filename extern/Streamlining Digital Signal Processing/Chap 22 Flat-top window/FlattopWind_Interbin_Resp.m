% Filename: FlattopWind_Interbin_Resp.m
%
%  Demonstrates the performance of the various flat-top window 
%  coefficients described in the March 2011 IEEE "Tips & Tricks" article
%  "Reducing FFT Scalloping Loss Errors Without Multiplication".
%
%   Richard Lyons, Jan. 2011

clear, clc
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Scan a sinewave from one bin center to the next bin center,
% and measure various "scalloping" loss values.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	N = 128; % Number of time samples and number of freq points 
	FS = N; % Sample rate so that DFT resolution is one Hz.
	T = 0:1/FS:(N-1)/FS;
	
	StartBin = 19; % FFT bin index where freq sweep begins 
	
	for K = 1:N % Sweep a sinewave's freq across a single DFT bin ("StartBin").
        
		CNTR_FREQ = StartBin + (K-1)/N; % Scans from m=StartBin -to- m=StartBin + 1.
		SIG = cos(2*pi*CNTR_FREQ*T + pi/4);	% Current center freq of sinewave
		
		% Apply various windows to the sweeping input sinewave
			Sig_Flattop_Matlab = flattopwin(N)'.*SIG/0.2156; % Eliminate window gain loss
			Sig_Hann = hanning(N, 'periodic')'.*SIG;
			Sig_Hamm = hamming(N, 'periodic')'.*SIG;
			Sig_Rect = rectwin(N)'.*SIG;
		
		% Measure various spectral magnitude peak values
			Max_Mag_Flattop_Matlab(K) = max(abs(fft(Sig_Flattop_Matlab, N)));
			Max_Mag_Hann(K) = max(abs(fft(Sig_Hann, N)));
			Max_Mag_Hamm(K) = max(abs(fft(Sig_Hamm, N)));
			Max_Mag_Rect(K) = max(abs(fft(Sig_Rect, N)));
            
            
	% Compute freq-domain convolutions using various SFT3F coeffs
        Spec = fft(SIG, N);
        Spec_Mag = abs(Spec);
        Max_Spec_Mag = max(Spec_Mag);
        m = find(Spec_Mag==Max_Spec_Mag); m = m(1);
	
        SFT3F_Flt_Point_Max_Mag(K) = abs(Spec(m) + ...
                                        -0.94247*(Spec(m-1) + Spec(m+1)) + ...
                                         0.44247*(Spec(m-2) + Spec(m+2)));
	
        SFT3F_4_Bit_Max_Mag(K) = abs(Spec(m) + ...
                                        -(1/2+1/4+1/8+1/16)*(Spec(m-1) + Spec(m+1)) + ...
                                         (1/4+1/8+1/16)*(Spec(m-2) + Spec(m+2)));
     
        SFT3F_8_Bit_Max_Mag(K) = abs(Spec(m) + ...
                                        -(1/2+1/4+1/8+1/16+1/256)*(Spec(m-1) + Spec(m+1)) + ...
                                         (1/4+1/8+1/16+1/256)*(Spec(m-2) + Spec(m+2)));
	end % end "K" loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	figure(1), clf
	plot(Max_Mag_Rect/max(Max_Mag_Rect), '-g')
	hold on
	plot(Max_Mag_Hann/max(Max_Mag_Hann), '--b')
	plot(Max_Mag_Hamm/max(Max_Mag_Hamm), ':r')
	plot(Max_Mag_Flattop_Matlab/max(Max_Mag_Flattop_Matlab), '-k')
    hold off
	title('Normalized bin-to-bin magnitude')
	axis([0, N, 0.6, 1.05])
    legend('Rect', 'Hanning', 'Hamming', 'Matlab Flat-top')
	grid on, zoom on
	
	figure(2)
	plot(SFT3F_Flt_Point_Max_Mag/(N/2), ':k')
	hold on
	plot(SFT3F_4_Bit_Max_Mag/(N/2), '--r')
	plot(SFT3F_8_Bit_Max_Mag/(N/2), '-b')
	hold off
	legend('SFT3F Flt Point', 'SFT3F 4 Bit', 'SFT3F 8 Bit')
	grid on, zoom on
