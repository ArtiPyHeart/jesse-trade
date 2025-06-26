  function [H, M] = downsample_design(fs_out, fs_in)
%
%DOWNSAMPLE_DESIGN - designes downsampling polyphase filter
%   DOWNSAMPLE_DESIGN computes the filter matrix for downsampling.  fs_in is
%   the input sample rate, fs_out is the output sample rate.  The filter
%   design is controlled by parameters R, the filter span and the M the
%   decimation factor which sets the resampling resolution.  These factors
%   are under the control of the user.
%
%     Douglas W. Barker
%     ITT Corporation, Advanced Engineering & Sciences
%     141 National Business Parkway, Suite 200
%     Annapolis Junction, MD  20701
%     Voice:  301-497-9900 x170
%     Fax:    301-497-0207
%     email:  doug.barker@itt.com
%


% Filter design parameters
  R = 16;                   % Filter span (in 1/fs_in time units)
  M = 512;                  % Decimation factor

% Compute the time-point matrix T (in 1/(L_prime*fs_in) = 1/(M*fs_out) time units)
  L_prime = M*fs_out/fs_in; % Non-integer interpolation factor
  L = ceil(L_prime);        % Required number of phases

  T = zeros(R, L);
  for r = 0 : R-1
    for c = 0 : L-1
      T(r+1, c+1) = r*L_prime + c;
    end;
  end;

% Compute filter coefficient matrix
  fc = 1/M;
  win = 0.54 - 0.46 * cos(2*pi*T/(L_prime*R-1));    % Hamming window
  hd = fc*sinc(fc*(T - (L_prime*R-1)/2));           % Desired response
  H = win .* hd;                                    % Window the desired response

% Plot the segmented impulse response
  figure(1);
  plot(T.'/L_prime, H.', '.');
  title('Anti-aliasing filter impulse response');
  xlabel('Time - t/t_a_d_c ');
  ylabel('Magnitude');
  grid on;

% Reverse taps
  H = flipud(H);

% Compute the equivalent frequency response, to do this, h is computed with
%  regular tap spacing, h is therefore recomputed here
  t = [0 : floor(R*L_prime)];                       % Time point vector
  fc = 1/M;
  win = 0.54 - 0.46 * cos(2*pi*t/(L_prime*R-1));    % Hamming window
  hd = fc*sinc(fc*(t - (L_prime*R-1)/2));           % Desired response
  h = win .* hd;

% Evaluate frequency response only over the first few Nyquist zones
  w = [0 : 4095]*pi*(4/M)/4096;
  Hf = freqz(h, 1, w);
  f = w*M*fs_out/(2*pi);

% Plot the frequency response
  figure(2);
  plot(f/fs_out, 20*log10(abs(Hf)));
  title('Anti-aliasing filter frequency response');
  xlabel('Frequency - f/f_s');
  ylabel('Magnitude - dB');
  axis([0, 2, -120, +10]);
  grid on;






