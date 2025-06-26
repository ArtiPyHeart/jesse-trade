
function  [A_of_k, B_of_k] = QuadFilterCoeffs_Turner(w1, w2, a, N)
% [A_of_k, B_of_k] = QuadFilterCoeffs_Turner
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
% Compute A_of_k (in-phase) and B_of_k (quad-phase) 
%    coefficients from Eqs. (9) & (10).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    k = 0:N-1;                % Index of A_of_k coefficients
    t = 2*pi*(k-(N-1)/2);      % Time-domain variable vector for Eq. (10)
    A_numer = 2*pi*pi*cos(a*t);       % Numerator factor of A_of_k
    A_denom = t.*(4*a*a*t.*t-pi*pi);  % Denominator
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
    