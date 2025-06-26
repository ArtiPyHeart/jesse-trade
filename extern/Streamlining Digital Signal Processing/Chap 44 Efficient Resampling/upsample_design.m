  function H = upsample_design
%
%UPSAMPLE_DESIGN - designs upsample filter
%   UPSAMPLE_DESIGN - designs a filter for use in upsampling, where the
%   output sample rate is higher than the input sample rate.  The
%   anti-aliasing low pass filter is designed to have a cut-off frequency
%   equal to the boundary of the first and second Nyquist zones.  Notice
%   that the sample rate ratio is not a factor in the design of the filter.
%
%     Douglas W. Barker
%     ITT Corporation, Advanced Engineering & Sciences
%     141 National Business Parkway, Suite 200
%     Annapolis Junction, MD  20701
%     Voice:  301-497-9900 x170
%     Fax:    301-497-0207
%     email:  doug.barker@itt.com
%


% Design parameters
  R = 16;     % Filter span
  L = 512;    % Number of phases

% Design the anti-aliasing low pass filter filter
  Nt = R*L;                           % Total number of taps
  fc = 1/L;                           % Cut-off frequency (choose edge of first Nyquist zone)
  h = fir1(Nt-1, fc, hamming(Nt));    % Design low pass using window technique

% Sort into matrix form, R taps/L phases
  H = zeros(R,L);
  for r = 0 : R-1
    H(r+1,:) = h(r*L+1 : (r+1)*L);
  end;

% Calculate the frequency response
  w = [0 : 1023]*pi*(5/L)/1024;   % Evaluate response over the first 5 Nyquist zones
  Hf = freqz(h, 1, w);
  f = w*L/(2*pi);
  t = [0 : length(h)-1]'/L;
  T = reshape(t, size(H'));

% Plot the impulse response
  figure(1);
  plot(T, H'*L, '.');
  title('Anti-aliasing filter impulse response');
  ylabel('Magnitude');
  xlabel('Time - t/T_s');
  grid on;

% Plot the frequency response
  figure(2);
  plot(f, 20*log10(abs(Hf)));
  title('Anti-aliasing filter frequency response');
  xlabel('Frequency - f/f_s_,_i_n');
  ylabel('Magnitude - dB');
  grid on;
