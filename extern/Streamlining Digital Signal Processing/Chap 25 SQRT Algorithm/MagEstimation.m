
%  filename:  MagEstimation.m
%
%  MATLAB code in conjunction with:
% 
%     M. Allie and R. Lyons, "A Root of Less Evil", IEEE Signal 
%     Processing Magazine, DSP Tips & Tricks column, Vol. 22,
%     No. 2, March 2005. 
%
%  Compares the performance of three "magnitude estimation algorithms:
%
%  1) Lyons' "15*Max/16 + 15*Min/32"
%     square root algo ("Understanding Digital Signal
%     Processing, 2nd Edition", pp. 482)
%
%  2) algo from: "Linear Approx to Sqrt(x^2+y^2)"
%     article in IEEE Trans. on Audio and Electroacoustics,
%     December 1973, by Filip.  You have hardcopy of this article.
%
%  3) the "Value Set 4" algo from: W. Adams, and J. Brady,  
%    "Magnitude Approximations for Microprocessor Implementation", 
%    IEEE Micro, pp. 27-31, Oct. 1983.  You have hardcopy 
%    of this article.
%
%   Copyright: Richard Lyons, August 2004.  
%   Use this code for whatever you wish, but please leave
%   these comment lines unchanged.

clear

for K = 1:91
   Phase(K) = K-1; % Phase angles in degrees (0 -to- 90 degrees)

   I = cos(Phase(K)*pi/180); % Real part
   Q = sin(Phase(K)*pi/180); % Imag part
   
   MaxIQ = max(abs(I),abs(Q)); % Get max and min of I & Q
   MinIQ = min(abs(I),abs(Q));
   
   PhaseRotated(K) = atan2(MinIQ,MaxIQ)*180/pi;
   Ratio(K) = MinIQ/MaxIQ;
   
   LyonsEstimate(K) = 15*(MaxIQ + MinIQ/2)/16; % Lyons
    %LyonsEstimate(K) = 31*MaxIQ/32 + 3*MinIQ/8; % Blankenship
    %LyonsEstimate(K) = MaxIQ + 0.5*MinIQ; % Robertson
    %LyonsEstimate(K) = MaxIQ + 0.25*MinIQ; 
    
    %if PhaseRotated(K) <= 22.5
   if Ratio(K) <= 1/4 
      AdamsEstimate(K) = MaxIQ + 0.0*MinIQ; % Here's the algorithm
   else
      AdamsEstimate(K) = (7/8)*MaxIQ + (1/2)*MinIQ; % Here's the algorithm
   end
   if Ratio(K) <= sqrt(2)-1 
      FilipEstimate(K) = 0.99*MaxIQ + 0.197*MinIQ; % Here's the algorithm
   else
      FilipEstimate(K) = 0.84*MaxIQ + 0.561*MinIQ; % Here's the algorithm
   end
   ErrorLyons(K) = 100*abs(LyonsEstimate(K)-1); % Absolute error
   ErrorAdams(K) = 100*abs(AdamsEstimate(K)-1); % Absolute error
   ErrorFilip(K) = 100*abs(FilipEstimate(K)-1); % Absolute error
   
end

figure(1)

%   ************* Plot results  *****************
subplot(2,1,1), plot(Phase,LyonsEstimate,'b'), grid on, zoom on
title('Est. magnitudes: Blue-Lyons, Black-Filip, Green-Adams')

hold on
plot(Phase,AdamsEstimate,'g'), grid on, zoom on
plot(Phase,FilipEstimate,'k'), grid on, zoom on
hold off
text(22, 0.94, 'Correct magnitude = 1.')

subplot(2,1,2)
plot(Phase,ErrorLyons,'b'), grid on, zoom on
hold on
plot(Phase,ErrorAdams,'g'), grid on, zoom on
plot(Phase,ErrorFilip,'k'), grid on, zoom on
hold off
title('Absolute error')
xlabel('Phase angle of complex vector (degrees)')
ylabel('Percent error')

disp(' '), disp(' ')
disp('Max Abs percent error'), disp('     Lyons     Filip    Adams')
[max(ErrorLyons), max(ErrorFilip), max(ErrorAdams)]
disp('Average Abs percent error'), disp('     Lyons     Filip     Adams')
[mean(ErrorLyons), mean(ErrorFilip), mean(ErrorAdams)]
disp('Stand. deviation of percent error'), disp('     Lyons     Filip     Adams')
[std(ErrorLyons), std(ErrorFilip), std(ErrorAdams)]
disp('Variance of percent error'), disp('     Lyons     Filip     Adams')
[std(ErrorLyons)^2, std(ErrorFilip)^2, std(ErrorAdams)^2]

