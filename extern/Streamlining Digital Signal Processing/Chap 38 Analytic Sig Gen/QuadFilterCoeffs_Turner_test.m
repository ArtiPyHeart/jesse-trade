
% Filename: QuadFilterCoeffs_Turner_test.m

%  Tests the "QuadFilterCoeffs_Turner()" and 
%  "QuadFilterCoeffs_Turner_plots()" functions associated with 
%  the article: "An Efficient Analytic Signal Generator"
%  by Clay S. Turner.
%
%  Creates a real-valued time signal, containing three sinusoids, 
%  and then turns that signal into a complex-valued analytic signal.
%
%  Filters' parameters: "w1", "w2" "a", and "N" are defined
%  somewhere near line 60 below. 
%
%    w1: beginning frequency of filters' passband (-3dB point).
%       "w1" must be in the range of 0 -to- 0.5. 
%    w2: end frequency of filters' passband (-3dB point).
%       "w2" must be in the range of 0 -to- 0.5, and greater than w1. 
%    a: half the filters' desired transition-region width.
%       "a" must be in the range of 0 -to- 0.5. 
%    N: number of filter taps.
%    A_of_k: the computed in-phase filter's coefficients.
%    B_of_k: the computed quadrature-phase filter's coefficients.
%
% Code by Richard (Rick) Lyons.  Sept. 2008.
% Use this code however you wish, but please do not change 
% the above comment lines. 

clear all, clc
        
% Define a real-valued bandpass test signal
	Num_Samples = 256;            % Number of time-domain samples
	Fs = 1000;                    % Sample rate in Hz.
	Time = (1/Fs)*(0:Num_Samples-1); % Time vector
	Freq_1 = 100;                 % 1st spec component freq 
	Freq_2 = 150;                 % 2nd spec component freq 
	Freq_3 = 200;                 % 3rd spec component freq 
	
	Sig_In = cos(2*pi*Freq_1*Time)+...
             0.75*cos(2*pi*Freq_2*Time)+...
             0.5*cos(2*pi*Freq_3*Time);   % Sum of three sinusoids
         
	% Add some noise
        Sig_In = Sig_In + 0.3*randn(1, Num_Samples); 
    
    % Plot spec mag of real-valued test signal
		Spec_Sig_In = fft(Sig_In);
		Spec_Sig_In_Mag = fftshift(abs(Spec_Sig_In));
		Freq_Axis = -Num_Samples/2:Num_Samples/2-1; % Plot zero Hz in the middle
		Freq_Axis = Freq_Axis*Fs/Num_Samples;  % Normalize to Fs sample rate
        
        figure(1)
		subplot(2,1,1)
        plot(Freq_Axis, Spec_Sig_In_Mag)
		xlabel(['Hertz (Fs = ',num2str(Fs),' Hz)'])
		ylabel('Input Sig Spec Mag')
        grid on, zoom on
    
% Now create an analytic signal based on real-valued "Sig_In" signal
	% Define the filters
		w1 = 0.05;   w2 = 0.45;   a = 0.05; % Example in Tips & Tricks column
		NumTaps = 50;              % Number of filter taps, "N", in Eq. (10)
		
		% Compute the filters' coefficients by choosing 
        %   one of the two functions below.
		[A_coeffs, B_coeffs] = QuadFilterCoeffs_Turner(w1, w2, a, NumTaps);
		[A_coeffs, B_coeffs] = QuadFilterCoeffs_Turner_plots(w1, w2, a, NumTaps);
        
	% Pass real-valued "Sig_In" signal through the filters and create the
	% final complex output signal.
        i_of_n = filter(A_coeffs, 1, Sig_In);
        q_of_n = filter(B_coeffs, 1, Sig_In);
        Cmplx_Sig_Out = i_of_n +j*q_of_n;
        
            % Plot spec mag of complex-valued "Cmplx_Sig_Out" signal
			Spec_Sig_Out = fft(Cmplx_Sig_Out);
			Spec_Sig_Out_Mag = fftshift(abs(Spec_Sig_Out));
            
			figure(1)
            subplot(2,1,2)
            plot(Freq_Axis, Spec_Sig_Out_Mag)
		    xlabel(['Hertz (Fs = ',num2str(Fs),' Hz)'])
		    ylabel('Output Sig Spec Mag')
			grid on, zoom on

    