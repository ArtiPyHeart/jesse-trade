%  MATLAB code in conjunction with:
% 
%     M. Allie and R. Lyons, "A Root of Less Evil", IEEE Signal 
%     Processing Magazine, DSP Tips & Tricks column, Vol. 22,
%     No. 2, March 2005. 
%
%  Models the Nonlinear IIR Filter (NIIRF) square root algorithm
%  from Reference [1] in the above article.
%
%    The acceleration factor "Beta" is found from a 
%    look-up table.
%

clear
NumPts = 32768;  % Change to 16384 for Student version of MATLAB;
x=[0.25:1/NumPts:1];
P0 = 2/3*x+.354167;
for i=[1:length(x)]
    if ((x(i)>=0.25) & (x(i)<0.3125))
        Beta(i) = 0.961914;
    elseif ((x(i)>=0.3125) & (x(i)<0.375))
        Beta(i) = .840332;
    elseif ((x(i)>=0.375) & (x(i)<0.4375))
        Beta(i) = .782715;
    elseif ((x(i)>=0.4375) & (x(i)<0.5))
        Beta(i) = .734869;
    elseif ((x(i)>=0.5) & (x(i)<0.5625))
        Beta(i) = .691406;
    elseif ((x(i)>=0.5625) & (x(i)<0.625))
        Beta(i) = .654297;
    elseif ((x(i)>=0.625) & (x(i)<0.6875))
        Beta(i) = .622070;
    elseif ((x(i)>=0.6875) & (x(i)<0.75))
        Beta(i) = .595215;
    elseif ((x(i)>=0.75) & (x(i)<0.8125))
        Beta(i) = .573731;
    elseif ((x(i)>=0.8125) & (x(i)<0.875))
        Beta(i) = .556152;
    elseif ((x(i)>=0.875) & (x(i)<0.9375))
        Beta(i) = .516113;
    elseif ((x(i)>=0.9375) & (x(i)<=1.0))
        Beta(i) = .502930;
    end
end
P1 = Beta.*(x-(P0.*P0))+P0;
P2 = Beta.*(x-(P1.*P1))+P1;

er=sqrt(x)-P2;
per=er./sqrt(x);
s_per=per*per';
ms_per=s_per/length(x);

ter=sqrt(x)-P1;
tper=ter./sqrt(x);
ts_per=tper*tper';
tms_per=ts_per/length(x);

figure(1)
plot(x,per*100,'b')
grid on
title('Mikami - Error/SQRT(x),  Beta from LUT');
xlabel('x')
ylabel('Percent')
v=axis;
axis([0,1,v(3),v(4)]);
set(gca,'xtick',[0,2/16,4/16,6/16,8/16,10/16,12/16,14/16,16/16,]);
set(gca,'XTickLabel',{'0','1/8','2/8','3/8','4/8','5/8','6/8','7/8 ','1',});

disp(sprintf(' Floating Mikami Algorithm 2 iterations: Values in percent except mse'))
disp(sprintf(' Mean Squared Error = %e     Maximum Error = %e',ms_per,max(abs(per)*100)))
disp(sprintf(' Average Error = %e     Variance = %e',mean(abs(per))*100, std(per*100)^2))
disp(' ')
disp(sprintf(' Floating Mikami Algorithm 1 iteration: Values in percent except mse'))
disp(sprintf(' Mean Squared Error = %e     Maximum Error = %e',tms_per,max(abs(tper)*100)))
disp(sprintf(' Average Error = %e     Variance = %e',mean(abs(tper))*100, std(tper*100)^2))


