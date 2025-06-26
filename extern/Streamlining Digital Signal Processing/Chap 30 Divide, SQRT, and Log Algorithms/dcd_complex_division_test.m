function []=dcd_complex_division_test

ar=3.6; ai=3.20; br=2.10; bi=3.80; 
% ar=3.6; ai=3.40; br=2.10; bi=4.10; 

[xr,xi,k,Trajectory]=dcd_complex_division(ar,ai,br,bi,16,18,64);

TrueValue=(br+j*bi)/(ar+j*ai);
fprintf('true value :       %e + j %e\n', real(TrueValue), imag(TrueValue));
fprintf('algorithm output : %e + j %e, obtained after %i iterations\n', ...
        xr, xi, k+1);

for kk=1:k+1,
 AbsoluteError(kk)=abs(Trajectory(1,kk)+j*Trajectory(2,kk)-TrueValue);
 fprintf('%d %e +j %e, %e\n', ...
         kk, Trajectory(1,kk), Trajectory(2,kk), AbsoluteError(kk));
end;

MinX=min(Trajectory(1,:));
MaxX=max(Trajectory(1,:));
MinY=min(Trajectory(2,:));
MaxY=max(Trajectory(2,:));

figure(1);
plot(Trajectory(1,:),Trajectory(2,:),'linewidth',1.5);
grid;
xlabel('real part');
ylabel('imaginary part');
axis([MinX-0.1*(MaxX-MinX), MaxX+0.1*(MaxX-MinX),...
      MinY-0.1*(MaxY-MinY), MaxY+0.1*(MaxY-MinY)]);
title('complex DCD algorithm');

figure(2);
semilogy(1:k+1,AbsoluteError,'k-',1:k+1,0.75*10.^(-0.2*((1:k+1)-2)),'k:','linewidth',1.5); grid on

xlabel('succesfull iterations');
ylabel('abs(x_n-x_{true})');
title('convergence of the complex DCD algorithm');

% print -deps -f1 dcd_complex_division_test_fig1.eps
% print -djpeg90 -f1 dcd_complex_division_test_fig1.jpg
% print -deps -f2 dcd_complex_division_test_fig2.eps
% print -djpeg90 -f2 dcd_complex_division_test_fig2.jpg

