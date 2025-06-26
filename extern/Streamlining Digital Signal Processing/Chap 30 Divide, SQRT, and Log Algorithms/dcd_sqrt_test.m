
c=2;
[x,k,Trajectory]= dcd_sqrt(c,2,32,100);
TrueX=sqrt(c);

for kk=1:k+1,
 AbsoluteError(kk)=norm(Trajectory(1,kk)-TrueX);
 fprintf('%d %e %e\n', kk, Trajectory(1,kk), AbsoluteError(kk));
end;

figure(1);
plot(1:k+1,Trajectory(1,1:k+1)); grid;
xlabel('n');
ylabel('x_n');
title('convergence of the sqrt DCD algorithm');

figure(2);
semilogy(1:k+1,AbsoluteError,'k-',1:k+1,10.^(-(1:k+1)/2),'k:','linewidth',1.5); grid;
xlabel('n');
ylabel('abs(x_n-x_{true})');
title('convergence of the sqrt DCD algorithm');

% print -djpeg90 -f2 ExponentialConvergenceMatrixCase.jpg

% print -deps    -f2 dcd_sqrt_test_fig2.eps
% print -djpeg90 -f2 dcd_sqrt_test_fig2.jpg

