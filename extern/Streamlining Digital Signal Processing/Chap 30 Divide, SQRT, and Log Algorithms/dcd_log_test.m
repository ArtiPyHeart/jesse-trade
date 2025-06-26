x=5.0;
[logx,k,Trajectory]=dcd_log(x,40,100);
True_logx=log(x);

for kk=1:k+1,
 AbsoluteError(kk)=norm(Trajectory(2,kk)-True_logx);
 fprintf('%2d %18.11e %18.11e\n', kk, Trajectory(2,kk), AbsoluteError(kk));
end;

figure(1);
plot(1:k+1,Trajectory(2,1:k+1)); grid;
xlabel('n');
ylabel('y_n');
title('convergence of the log algorithm');

figure(2);
semilogy(1:k+1,AbsoluteError,'k-','linewidth',1.5); grid;
xlabel('n');
ylabel('abs(x_n-x_{true})');
title('convergence of the log algorithm');