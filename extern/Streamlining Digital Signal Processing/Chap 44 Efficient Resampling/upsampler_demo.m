  function upsampler_demo
%
%UPSAMPLER_DEMO - demonstrates the operation of the upsampling algorithm
%   UPSAMPLER_DEMO - implements figure 1 in the paper.  The user may change
%   the input sample rate fs_in, the test tone frequency fc, and the signal to
%   noise ratio cnr if desired.  The design of the resampling filter is
%   done in the function 'UPSAMPLER_DESIGN'
%
%     Douglas W. Barker
%     ITT Corporation, Advanced Engineering & Sciences
%     141 National Business Parkway, Suite 200
%     Annapolis Junction, MD  20701
%     Voice:  301-497-9900 x170
%     Fax:    301-497-0207
%     email:  doug.barker@itt.com
%

% System parameters
  fs_out = 100e6;         % 100 MHz D/A converter sample rate [sps]
  Nnco = 32;              % Number of bits in the NCO
  Nfft = 4096;            % Number of points in FFT's

% Input test signal parameters
  Ns = 1e4;               % Number of input samples in simulation
  fs_in = 7.913e6;        % Input sample rate [sps]
  fc = 531e3;             % Test tone frequency [Hz]
  cnr = 40;               % Carrier to noise ratio of input signal [dB]

% Design the resampling filter
  H = upsample_design;
  [R, L] = size(H);       % Get filter span and number of phases
  H = L*H;                % Scale coefficients

% Synthesize a test signal (keep things simple, use real lowpass signal)
  x = sqrt(2)*cos(2*pi*fc*[0 : Ns-1]'/fs_in);   % Ps = 1.000
  noise = randn(Ns, 1)*(10^(-cnr/20));              % Generate WGN, Pn = -cnr dB
  x = x + noise;                                    % Add the signal and noise

% Compute and plot power spectral density
  [Xf, f] = pwelch(x, hanning(Nfft), [], [], fs_in);
  figure(3);
  plot(f/1e6, 10*log10(Xf*fs_in/Nfft));
  title('Power spectral density of input test signal');
  xlabel('Frequency - MHz');
  ylabel('Amplitude - dB');
  axis([0, fs_in/2/1e6, -100, +10]);
  grid on;

% Implementation of Figure 1(a) in the paper
  phz_inc = round(pow2(Nnco)*fs_in/fs_out);   % NCO phase increment
  y = zeros(floor(Ns*fs_out/fs_in),1);        % Reserve storage for output signal
  nco = 0;                                    % Init the NCO
  n = 1;                                      % Input signal index

  for m = 1 : floor(Ns*fs_out/fs_in) - ceil(R*fs_out/fs_in)  % m is output signal index

    % Get next phase to compute (equation 3)
    phz = bitshift(nco, log2(L) - Nnco);

    % Calculate filter outputs
    y(m) = x(n+R-1 : -1 : n).' * H(:, phz+1);

    % Update the NCO
    nco = nco + phz_inc;

    % If NCO overflows, get a new sample into filter
    if nco >= pow2(Nnco)
      nco = nco - pow2(Nnco);
      n = n + 1;                % Push another sample in the TDL
    end;
  end;

% Plot the power spectral density of the resulting output signal
  [Yf, f] = pwelch(y, hanning(Nfft), [], [], fs_out);
  figure(4);
  plot(f/1e6, 10*log10(Yf*fs_out/Nfft));
  title('Power spectral density upsampled output');
  ylabel('Power Density - dB');
  xlabel('Frequency - MHz');
  axis([0, +fs_out/2/1e6, -100, +5]);
  grid on;


