%  MATLAB code in conjunction with:
% 
%     M. Allie and R. Lyons, "A Root of Less Evil", IEEE Signal 
%     Processing Magazine, DSP Tips & Tricks column, Vol. 22,
%     No. 2, March 2005. 
%
%  Models the Nonlinear IIR Filter (NIIRF) square root algorithm
%  from Reference [1] in the above article.
%
%    The acceleration factor "Beta" is set to a constant 
%    for all values of input "x".
%

clear
NumPts = 32768;  % Change to 16384 for Student version of MATLAB
x=[0.25:1/NumPts:1];
P0 = 2/3.*x+.354167;

Beta = 0.633; % Two-iteration Beta
P1 = Beta*(x-(P0.*P0))+P0;
P2 = Beta*(x-(P1.*P1))+P1;
er=sqrt(x)-P2;
per=er./sqrt(x);
s_per=per*per';
ms_per=s_per/length(x);

Beta = 0.64; % One-iteration Beta
P1 = Beta*(x-(P0.*P0))+P0;
ter=sqrt(x)-P1;
tper=ter./sqrt(x);
ts_per=tper*tper';
tms_per=ts_per/length(x);

figure;
plot(x,per*100,'k',x,tper*100,'r')
grid on
title('Mikami - Error/SQRT(x), Beta = Const., 2 iter = black, 1 iter = red');
xlabel('x')
ylabel('Percent')
v=axis;
axis([0,1,v(3),v(4)]);
set(gca,'xtick',[0,2/16,4/16,6/16,8/16,10/16,12/16,14/16,16/16,]);
set(gca,'XTickLabel',{'0','1/8','2/8','3/8','4/8','5/8','6/8','7/8 ','1',});

disp(sprintf(' Mikami Algorithm 2 iterations: Beta = %g  Values in percent except mse ',Beta))
disp(sprintf(' Mean Squared Error = %e     Maximum Error = %e',ms_per,max(abs(per)*100)))
disp(sprintf(' Average Error = %e     Variance = %e',mean(abs(per))*100, std(per*100)^2))
disp(' ')
disp(sprintf(' Mikami Algorithm 1 iteration: Beta = %g  Values in percent except mse',Beta))
disp(sprintf(' Mean Squared Error = %e     Maximum Error = %e',tms_per,max(abs(tper)*100)))
disp(sprintf(' Average Error = %e     Variance = %e',mean(abs(tper))*100, std(tper*100)^2))

