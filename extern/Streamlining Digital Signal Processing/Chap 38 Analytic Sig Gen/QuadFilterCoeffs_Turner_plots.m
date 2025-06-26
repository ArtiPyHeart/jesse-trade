
function  [A_of_k, B_of_k] = QuadFilterCoeffs_Turner_plots(w1, w2, a, N)
% [A_of_k, B_of_k] = QuadFilterCoeffs_Turner_plots
%
% Computes the coefficients for the quadrature filters
% described by article: "An Efficient Analytic Signal Generator"
% by Clay S. Turner.
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
% This function computes the filters' coefficients and plots 
% the performance of the quadrature filters.
%
% Code by Richard (Rick) Lyons.  Sept. 2008.
% Use this code however you wish, but please do not change 
% the above comment lines. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin the function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % A little error checking
    if w1<a | w1>0.5-a
        beep, disp(' '), disp(' ')
        error('w1 must be in the range: a <= w1 <= 0.5-a');
    end
    if w2<=w1
        beep, disp(' '), disp(' ')
        error('w2 must be greater than w1');
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute and plot desired filter mag response [Eq. (1)]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	NumFreqPts = 200;                          % Num of freq samples for plotting.
    FreqSegLen_1 = floor((w1-a)*2*NumFreqPts); % Length of initial zeros
    FreqSegLen_2 = floor(2*a*2*NumFreqPts);    %  1st transition region
    FreqSegLen_3 = floor(((w2-a)-(w1+a))*2*NumFreqPts); % Passband
    FreqSegLen_4 = floor(2*a*2*NumFreqPts);    %  2nd transition region
    FreqSegLen_5 = floor((0.5-(w2+a))*2*NumFreqPts); % Final zeros

	w = (0:NumFreqPts-1)/(2*NumFreqPts);       % Freq axis samples (0 -to- 0.5)
	Mag(1:FreqSegLen_1) = zeros(1, FreqSegLen_1); % Initial zero samples 
	Mag = [Mag, (sin((pi/(4*a))*(w(FreqSegLen_1+1:FreqSegLen_1...
            +FreqSegLen_2)-(w1-a)))).^2];      % 1st transition region
	Mag = [Mag, ones(1, FreqSegLen_3)];        % Unity valued passband
	Mag = [Mag, (cos((pi/(4*a))*(w(FreqSegLen_1+FreqSegLen_2+FreqSegLen_3+1:FreqSegLen_1...
            +FreqSegLen_2+FreqSegLen_3+FreqSegLen_4)-(w2-a)))).^2]; % 2nd transition region
	Mag = [Mag, zeros(1, FreqSegLen_5)];       % Final zero samples
	Mag(NumFreqPts) = 0;                        % Fill in final zeros (if necessary)
	
	figure(51)
	plot(w, Mag, '-b')
	ylabel('Desired Mag (Linear)'), xlabel('Freq (x Fs)')
	axis([0, 0.5, 0, 1.1]), grid on, zoom on
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Evaluate, and plot, Eq. (9) [rotated in-phase filter imp. resp.]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    t = linspace(-100, 100, 201);
    A_numer = 2*pi*pi*cos(a*t);           % Numerator factor of A(t)
    A_denom = t.*(4*a*a*t.*t-pi*pi);      % Denominator
    A_of_t = (sin(w1*t+pi/4)-sin(w2*t+pi/4)).*(A_numer./A_denom);   % In-phase imp. resp.
    
        % Check for, and correct, indeterminate values in "A(t)" (eliminate the "NaNs")
        TempIndex = find(isnan(A_of_t)==1);  
        if length(TempIndex) > 0             % Are there any NaNs in A(t)?
			Angle_1 =  (pi/4)*((a+2*w2)/a);  % There are Nan's in A(t)
			Angle_2 =  (pi/4)*((a+2*w1)/a);   
			Angle_3 =  (pi/4)*((a-2*w1)/a);   
			Angle_4 =  (pi/4)*((a-2*w2)/a);   
            for K = 1:length(TempIndex)      % Correct A(t) for each NaN
                if t(TempIndex(K)) < -pi/(2*a)+0.1  % Is NaN at t=-pi/(2a)?
                    A_of_t(TempIndex(K)) = a*(sin(Angle_3)-sin(Angle_4));
                else, end
                if t(TempIndex(K)) > pi/(2*a)-0.1  % Is NaN at t=pi/(2a)?
                    A_of_t(TempIndex(K)) = a*(sin(Angle_1)-sin(Angle_2));
                else, end
                if abs(t(TempIndex(K))) < 0.001  % Is NaN at t=0?
                    A_of_t(TempIndex(K)) = sqrt(2)*(w2-w1);
                else, end
            end
        else end
    
    B_of_t = fliplr(A_of_t);  % Flip A(t), left to right
     
    figure(52)
    plot(t, A_of_t, '-b')
    hold on
    plot(t, B_of_t, ':k')
    hold off
    legend('A(t)', 'B(t)')
    ylabel('Amplitude'), xlabel('Time (seconds)')
    grid on, zoom on
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute A_of_k (in-phase) and B_of_k (quad-phase) 
%    coefficients from Eqs. (9) & (10).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    k = 0:N-1;                % Index of A_of_k coefficients
    t = 2*pi*(k-(N-1)/2);      % Time-domain variable vector for Eq. (10)
    A_numer = 2*pi*pi*cos(a*t);           % Numerator factor of A_of_k
    A_denom = t.*(4*a*a*t.*t-pi*pi);       % Denominator
    A_of_k = (sin(w1*t+pi/4)-sin(w2*t+pi/4)).*(A_numer./A_denom);   % In-phase imp. resp.
    
        % Check for, and correct, indeterminate values in "A_of_k" (eliminate the "NaNs")
        TempIndex = find(isnan(A_of_k)==1);  
        if length(TempIndex) > 0           % Are there any NaNs in A_of_k?
			Angle_1 =  (pi/4)*((a+2*w2)/a);   % There are Nan's in A_of_k
			Angle_2 =  (pi/4)*((a+2*w1)/a);   
			Angle_3 =  (pi/4)*((a-2*w1)/a);   
			Angle_4 =  (pi/4)*((a-2*w2)/a);   
            for K = 1:length(TempIndex) % Correct A_of_k for each NaN
                if t(TempIndex(K)) < -pi/(2*a)+0.1  % Is NaN at t=-pi/(2a)?
                    A_of_k(TempIndex(K)) = a*(sin(Angle_3)-sin(Angle_4));
                else, end
                if t(TempIndex(K)) > pi/(2*a)-0.1  % Is NaN at t=pi/(2a)?
                    A_of_k(TempIndex(K)) = a*(sin(Angle_1)-sin(Angle_2));
                else, end
                if abs(t(TempIndex(K))) < 0.0001  % Is NaN at t=0?
                    A_of_k(TempIndex(K)) = sqrt(2)*(w2-w1);
                else, end
            end
        else end
        
	B_of_k = fliplr(A_of_k); % Flip "A_of_k", left to right
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot final positive-freq mag response of the A_of_k coeffs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    FFT_size = 512;
    FreqResp_A_of_k = fft(A_of_k, FFT_size);
    FreqResp_B_of_k = fft(B_of_k, FFT_size);
        
    Mag_A_of_k = abs(FreqResp_A_of_k);
    Mag_A_of_k_dB = 20*log10(Mag_A_of_k/max(Mag_A_of_k));
    Phase_A_of_k = angle(FreqResp_A_of_k);
    Phase_B_of_k = angle(FreqResp_B_of_k);
    Phase_Diff = (Phase_A_of_k-Phase_B_of_k)*(180/pi);
    
        % Unwrap the phase difference;
        TempIndex = find(Phase_Diff < 0);
        Phase_Diff(TempIndex) = Phase_Diff(TempIndex) + 360;
    
    Freq_Axis = (0:FFT_size/2 -1)/FFT_size;
    
    figure(53)
    subplot(2,1,1)
    plot(k, A_of_k, '-bo', 'markersize', 4)
    hold on
    plot(k, B_of_k, ':ko', 'markersize', 4)
    hold off
	ylabel('Amplitude'), xlabel('k')
    legend('A_of_k', 'B-hat(k)')
    grid on, zoom on
    subplot(2,1,2)
    plot(w, Mag, 'y', 'linewidth', 4)
    hold on
    plot(Freq_Axis, Mag_A_of_k(1:FFT_size/2), 'r') % Pos freqs only
    hold off
	ylabel('Mag (Linear)'), xlabel('Freq (x Fs)')
	%axis([0, 0.5, 0, 1.1])
    legend('Desired A(t)', 'Actual A_of_k')
    grid on, zoom on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot full freq mag response of the analytic sig generator, and 
%   phase difference between A_of_k & B_of_k coeffs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    figure(54)

    % Plot full freq mag response of the analytic signal generator (ASG)
        FreqResp_ASG = fft(A_of_k+j*B_of_k, FFT_size);
        Mag__ASG = abs(fftshift(FreqResp_ASG));
        Mag__ASG_dB = 20*log10(Mag__ASG/max(Mag__ASG));
        % Threshold min values mag plot to make it look pretty
            Mag__ASG_dB(Mag__ASG_dB<-80) = -80;  % Min dB plot level
        Phase_ASG = angle(FreqResp_ASG);
        Freq_Axis_full = (-FFT_size/2 +1:FFT_size/2)/FFT_size;
        
        subplot(2,1,1)
        plot(Freq_Axis_full, Mag__ASG_dB, 'b')
            % Plot some markers
            hold on
            plot([w1, w1],[-20, 0],'-r')
            plot([w2, w2],[-20, 0],'-r')
            text(w1, -25, 'w1'), text(w2, -25, 'w2')
            hold off
        ylabel('|A_of_k| (dB)'), xlabel('Freq (x Fs)')
        text(-0.34, -26, 'Zoom in, if you wish')
        grid on, zoom on

        % Plot phase difference
        subplot(2,1,2)
        plot(Freq_Axis_full, fftshift(Phase_Diff), 'b')
        axis([-0.5, 0.5, 0, 350])
            % Plot some markers
            hold on
            plot([w1, w1],[60, 130],'-r')
            plot([w2, w2],[60, 130],'-r')
            text(w1, 150, 'w1'), text(w2, 150, 'w2')
            hold off
		ylabel('Phase-Diff. (Deg.)'), xlabel('Freq (x Fs)')
        text(0.15, 230, 'Zoom in, if you wish')
        grid on, zoom on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute & plot group delay of A_of_k
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [Group_Delay, W] = grpdelay(A_of_k, 1, FFT_size);
    
        % Look at positive freqs only
        Group_Delay = Group_Delay(1:FFT_size/2);
        W = W(1:FFT_size/2);
        Freq_Axis = (0:FFT_size/2 -1)/FFT_size;

    figure(55)
    plot(Freq_Axis, Group_Delay)
	ylabel('Group delay (samples)'), xlabel('Freq (x Fs)')
	axis([0, 0.5, (N-1)/2-4, (N-1)/2+4])
        % Plot some markers
        hold on
        plot([w1, w1],[(N-1)/2-0.2, (N-1)/2+0.2],'-r')
        plot([w2, w2],[(N-1)/2-0.2, (N-1)/2+0.2],'-r')
        text(w1, (N-1)/2-0.35, 'w1'), text(w2, (N-1)/2-0.35, 'w2')
        hold off
    text(0.07, (N-1)/2-1, 'Zoom in, if you wish')
    grid on, zoom on
    