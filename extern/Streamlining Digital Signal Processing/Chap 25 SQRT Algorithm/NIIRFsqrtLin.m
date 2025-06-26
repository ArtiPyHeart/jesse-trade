%  MATLAB code in conjunction with:
% 
%     M. Allie and R. Lyons, "A Root of Less Evil", IEEE Signal 
%     Processing Magazine, DSP Tips & Tricks column, Vol. 22,
%     No. 2, March 2005. 
%
%  Models the Nonlinear IIR Filter (NIIRF) square root algorithm
%  from Reference [1] in the above article.
%
%    The acceleration factor "Beta" is computed from a 
%    linear equation as a function of input "x".
%

clear
NumPts = 32768;  % Change to 16384 for Student version of MATLAB
x=[0.25:1/NumPts:1];
P0 = 2/3.*x+.354167;

Beta = 1.0688-0.61951*x;

P1 = Beta.*(x-(P0.*P0))+P0;
P2 = Beta.*(x-(P1.*P1))+P1;

er=sqrt(x)-P2;
per=er./sqrt(x);
s_per=per*per';
ms_per=s_per/length(x);

er1=sqrt(x)-P1;
per1=er1./sqrt(x);
s_per1=per1*per1';
ms_per1=s_per1/length(x);

figure;
plot(x,per*100,'b');
grid;
title('Mikami - Error/SQRT(x), Beta: Linear f(x)');
xlabel('x')
ylabel('Percent')
v=axis;
axis([0,1,v(3),v(4)]);
set(gca,'xtick',[0,1/16,2/16,3/16,4/16,5/16,6/16,7/16,8/16,9/16,10/16,11/16,12/16,13/16,14/16,15/16,16/16,]);
set(gca,'XTickLabel',{'0',' ','1/8',' ','2/8',' ','3/8',' ','4/8',' ','5/8',' ','6/8',' ','7/8 ',' ','1',});

disp(sprintf(' Float Mikami Algorithm 2 iterations Beta = Lin f(x) : Values in percent except mse'))
disp(sprintf(' Mean Squared Error = %e     Maximum Error = %e',ms_per,max(abs(per))*100))
disp(sprintf(' Average Error = %e     Variance = %e',mean(abs(per))*100, std(per*100)^2))
disp(' ')
disp(sprintf(' Float Mikami Algorithm 1 iteration Beta = Lin f(x) : Values in percent except mse'))
disp(sprintf(' Mean Squared Error = %e     Maximum Error = %e',ms_per1,max(abs(per1))*100))
disp(sprintf(' Average Error = %e     Variance = %e',mean(abs(per1))*100, std(per1*100)^2))

