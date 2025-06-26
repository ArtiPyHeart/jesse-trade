%  MATLAB code in conjunction with:
% 
%     M. Allie and R. Lyons, "A Root of Less Evil", IEEE Signal 
%     Processing Magazine, DSP Tips & Tricks column, Vol. 22,
%     No. 2, March 2005. 
%
%  Models the Newton-Raphson Inverse (NRI) square root algorithm.
%  Provides error statistics for both "two-iteration" and 
%  "one-iteration" vesrions.
%

clear
NumPts = 32768;  % Change to 16384 for Student version of MATLAB
x=[0.25:1/NumPts:1];
P0 = 1./(2/3*x+0.354167);
P1 = 0.5*P0.*(3-x.*P0.*P0);
P2 = 0.5*P1.*(3-x.*P1.*P1);
y=x.*P2;
er=sqrt(x)-y;
per=er./sqrt(x);
s_per=per*per';
ms_per=s_per/length(x);
y1=x.*P1;
er1=sqrt(x)-y1;
per1=er1./sqrt(x);
s_per1=per1*per1';
ms_per1=s_per1/length(x);
figure;
plot(x,per*100,'b')
title('Error/SQRT(x)')
xlabel('x')
ylabel('Percent')
grid on
v=axis;
axis([0,1,v(3),v(4)]);
set(gca,'xtick',[0,2/16,4/16,6/16,8/16,10/16,12/16,14/16,16/16,]);
set(gca,'XTickLabel',{'0','1/8','2/8','3/8','4/8','5/8','6/8','7/8 ','1',});
disp(' '), disp(' ')
disp(sprintf(' Floating Newton-Raphson Inverse Algorithm 2 iterations: Values in percent except MSE'))
disp(sprintf(' Mean Squared Error = %e     Maximum Error = %e',ms_per,max(abs(per)*100)))
disp(sprintf(' Average Error = %e     Variance = %e',mean(abs(per))*100, std(per*100)^2))
disp(' '), disp(' ')
disp(sprintf(' Floating Newton-Raphson Inverse Algorithm 1 iteration: Values in percent except MSE'))
disp(sprintf(' Mean Squared Error = %e     Maximum Error = %e',ms_per1,max(abs(per1)*100)))
disp(sprintf(' Average Error = %e     Variance = %e',mean(abs(per1))*100, std(per1*100)^2))

