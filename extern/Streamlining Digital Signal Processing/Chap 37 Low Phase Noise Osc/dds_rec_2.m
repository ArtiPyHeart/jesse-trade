

reg1=1;
reg2=0;

theta=0.0727*2*pi;
tht=theta*180/pi;
phi=atan(2.^-[0:9])*180/pi;
rr=-tht;
for nn=1:10
    cc(nn)=sign(rr);
    rr_sv(nn)=rr;
    rr=rr-cc(nn)*phi(nn);
end
theta_err=rr_sv(10)*pi/180;
scl=2^20;
del1=0;
del2=0;
for nn=1:1024
    
    vv(nn)=reg1+j*reg2;
    sum1=(cos(theta+theta_err)*reg1-reg2*sin(theta+theta_err));
    sum2=(cos(theta+theta_err)*reg2+reg1*sin(theta+theta_err));
    
    sum1_tmp=sum1+theta_err*sum2;
    sum2=sum2-theta_err*sum1;
    sum1=sum1_tmp;
    gn=(3-(sum1*sum1+sum2*sum2))/2;
    %gn=1;
    gain(nn)=gn;
    reg1=floor((1*del1+sum1*gn)*scl)/scl;
    del1=sum1*gn+del1-reg1;
    reg2=floor((1*del2+sum2*gn)*scl)/scl;
    del2=sum2*gn+del2-reg2;
end

figure(1)
subplot(2,1,1)
ww=kaiser(1024,15)';
ww=ww/sum(ww);

plot(-0.5:1/1024:0.5-1/1024,fftshift(20*log10(abs(fft(vv.*ww)))));
grid on
axis([-0.5 0.5 -150 5])

subplot(4,1,3)
plot(real(vv));
grid on
axis([0 1000 -1.2 1.2])

subplot(4,1,4)
plot(gain)
grid on
qq=axis;
axis([0 1000 qq(3) qq(4)])

figure(2)
subplot(2,1,1)
ww=kaiser(1024,18)';
ww=ww/sum(ww);

plot(-0.5:1/1024:0.5-1/1024,fftshift(20*log10(abs(fft(vv.*ww)))),'k');
grid on
axis([-0.5 0.5 -150 5])
title('Spectrum of Complex Sinusoid from Recusive CORDIC with AGC')
xlabel('Normalized Frequency')
ylabel('Log Magnitude (dB)')

subplot(2,1,2)
plot(gain-1,'k')
grid on
qq=axis;
axis([0 1000 qq(3) qq(4)])
title('\epsilon/2 = AGC Gain - 1')
xlabel('Time Index')
ylabel('Amplitude')

% 
% figure(2)
% ww=kaiser(1024,18)';
% ww=ww/sum(ww);
% 
% plot(-0.5:1/1024:0.5-1/1024,fftshift(20*log10(abs(fft(vv.*ww)))),'k');
% grid on
% axis([-0.5 0.5 -150 5])
% title('Spectrum of Complex Sinusoid from Recusive CORDIC with AGC')
% xlabel('Normalized Frequency')
% ylabel('Log Magnitude (dB)')


