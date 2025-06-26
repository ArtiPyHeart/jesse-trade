  function downsampler_demo
%
%DOWNSAMPLER_DEMO - demonstrates the operation of the downsampling algorithm
%   DOWNSAMPLER_DEMO - implements figure 1 in the paper.  The user may change
%   the input sample rate fs_in, the output sample rate, fs_out, the test tone
%   frequency fc, and the signal to noise ratio cnr if desired.
%
%     Douglas W. Barker
%     ITT Corporation, Advanced Engineering & Sciences
%     141 National Business Parkway, Suite 200
%     Annapolis Junction, MD  20701
%     Voice:  301-497-9900 x170
%     Fax:    301-497-0207
%     email:  doug.barker@itt.com
%


% Simulation parameters
  Ns = 1e5;               % Number of input samples in simulation
  fs_in = 100e6;          % 100 MHz input sample rate [sps]
  fs_out = 37.913e6;      % Output sample rate [sps]
  fc = 0.791e6;           % Test tone frequency [Hz]
  cnr = 40;               % Carrier to noise ratio of input signal [dB]

% Fixed parameters
  Nnco = 32;              % Number of bits in the NCO
  Nfft = 4096;            % Number of points in FFT's

% Design the downsampling filter
  [H, M] = downsample_design(fs_out, fs_in);
  [R, L] = size(H);       % Get filter span and number of phases
  H = L*H;                % Scale coefficients

% Synthesize a test signal (keep things simple, use real lowpass signal)
  x = sqrt(2)*cos(2*pi*fc*[0 : Ns-1]'/fs_in);     % Ps = 1.000
  noise = randn(Ns, 1)*(10^(-cnr/20));                % Generate WGN, Pn = -cnr dB
  x = x + noise;                                      % Add the signal and noise

% Compute and plot the power spectral density of the input signal
  [Xf, f] = pwelch(x, hanning(Nfft), [], [], fs_in);
  figure(3);
  plot(f/1e6, 10*log10(Xf*fs_in/Nfft));
  title('Power spectral density of input test signal');
  xlabel('Frequency - MHz');
  ylabel('Amplitude - dB');
  axis([0, fs_in/2/1e6, -100, +10]);
  grid on;

% Implementation of Figure 1(a) in the paper
  phz_inc = round(pow2(Nnco)*fs_out/fs_in);     % NCO phase increment
  y = zeros(floor(Ns*fs_out/fs_in),1);          % Reserve storage for output signal
  nco = 0;                                      % Init the NCO
  m = 1;                                        % Output signal index

  for n = R : Ns                          % n is input signal index

    % Resample signal every overflow
    if nco > pow2(Nnco)
      nco = nco - pow2(Nnco);
      phz = bitshift(nco, log2(M) - Nnco);
      y(m) = x(n : -1 : n-R+1).' * H(: , phz+1);
      m = m + 1;
    end;

    % Update the NCO
    nco = nco + phz_inc;
  end;

  y = y(1 : m-1);

% Plot the power spectral density of the output signal
  [Yf, f] = pwelch(y, hanning(Nfft), [], [], fs_out);
  figure(4);
  plot(f/1e6, 10*log10(Yf*fs_out/Nfft));
  title('Power spectral density downsampled output');
  ylabel('Power Density - dB');
  xlabel('Frequency - MHz');
  axis([0, +fs_out/2/1e6, -100, +5]);
  grid on;
