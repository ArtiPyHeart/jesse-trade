% DDS_Hybrid 5-stage CORDIC, Small Angle Correction, Gain Correction

N_CORD=input('enter number of Cordic Iterations     (try 6)     > ');
N_BITS=input('enter number of bits in multiplier    (try 16)    > ');
FREQ  =input('enter frequency (0->0.5) of sinusoid  (try 0.022) > ');
N_DAT =input('enter length of data sequence         (try 1024)  > ');
SD_ON =input('Sig-Del Dither on (1) or Sig-Del dither off (0)   > ');

reg1=1;
reg2=0;
del1=0;
del2=0;
accum=0;

tan_phi=(2.^-(0:N_CORD-1));
phi=atan(2.^-(0:N_CORD-1))/(2*pi);

scl2=prod(cos(2*pi*phi));

% number of bits in product
scl=2^N_BITS-1;

% normalized frequency f_c/f_s 
theta=FREQ;

for nn=1:N_DAT
    vv(nn)=reg1+j*reg2;
        accum=-theta;
% 5-stages of standard cordic        
        for kk=1:length(phi)
            sum1=(reg1+reg2*tan_phi(kk)*sign(accum));
            sum2=(reg2-reg1*tan_phi(kk)*sign(accum));
            reg1=sum1;
            reg2=sum2;
            accum=accum-phi(kk)*sign(accum);
        end
% standard cordic gain correction        
    reg1=reg1*scl2;
    reg2=reg2*scl2;
    
% small angle correction folded into cordic    
    sum1=(reg1+reg2*accum*2*pi*sign(accum));
    sum2=(reg2-reg1*accum*2*pi*sign(accum));

% gain correction matched to small angle correction term
    gn=(3-(sum1*sum1+sum2*sum2))/2;
    gain(nn)=gn;
    
% sigma-delta quantizer at output of gain correction I(n)
    reg1=floor((del1+sum1*gn)*scl)/scl;
    del1=SD_ON*(sum1*gn+del1-reg1);

    % sigma-delta quantizer at output of gain correction Q(n)
    reg2=floor((del2+sum2*gn)*scl)/scl;
    del2=SD_ON*(sum2*gn+del2-reg2);
    
end

figure(1)
subplot(2,1,1)
ww=kaiser(N_DAT,20)';
ww=ww/sum(ww);

plot(-0.5:1/N_DAT:0.5-1/N_DAT,fftshift(20*log10(abs(fft(vv(1:N_DAT).*ww)))));
hold on
plot([1 1]*FREQ,[-60 5],'r')
hold off
grid on
axis([-0.5 0.5 -160 5])
title('Spectrum of DDS : 5-Stage CORDIC and 1-Stage Small Angle Correction with AGC Gain Correction')

subplot(4,1,3)
plot(real(vv(1:200)));
grid on
axis([0 200 -1.2 1.2])
title('Real Part of DDS Output')

subplot(4,1,4)
plot(gain(1:200)-1)
grid on
qq=axis;
axis([0 200 qq(3) qq(4)])
title('Gain Correction tracking small angle correction and finite arithmetic')
