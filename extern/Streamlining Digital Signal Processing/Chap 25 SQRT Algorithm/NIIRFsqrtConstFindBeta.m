%  MATLAB code in conjunction with:
% 
%     M. Allie and R. Lyons, "A Root of Less Evil", IEEE Signal 
%     Processing Magazine, DSP Tips & Tricks column, Vol. 22,
%     No. 2, March 2005. 
%
%  Finds the optimum constant-acceleration factor "Beta" for 
%  both "one-iteration" and "two-iterations" of the Nonlinear 
%  IIR Filter (NIIRF) square root algorithm
%  from Reference [1] in the above article.
%

clear
NumPts = 32768;  % Change to 16384 for Student version of MATLAB
x=[0.25:1/NumPts:1];
P0 = 2/3.*x+.354167;

Beta = [.6:.001:.7];

mse=zeros(length(Beta),1);
ser1s=zeros(length(Beta),1);
sers=zeros(length(Beta),1);
for i=[1:length(Beta)]
    P1 = Beta(i)*(x-(P0.*P0))+P0;
    P2 = Beta(i)*(x-(P1.*P1))+P1;
    ser=sqrt(x)-P2;
    ser1=sqrt(x)-P1;
    sers(i)=ser*ser';
    ser1s(i)=ser1*ser1';
end
[ss,ii]=min(abs(sers));
Beta(ii);
[sss,iii]=min(abs(ser1s));
Beta(iii);

disp(' '), disp(' ')
disp(['Optimum Beta for one iteration is: ',num2str(Beta(iii))])
disp(['Optimum Beta for two iterations is: ',num2str(Beta(ii))])
disp(' ')

