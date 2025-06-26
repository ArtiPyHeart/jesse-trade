
function [x,k,Trajectory]=dcd_real_division(a,b,h_init,Mb,Nu)
% [x,k,Trajectory]=dcd_real_division(a,b,H,Mb,Nu)
% multiplication-free division of b by a 
% using the algorithm proposed by Liu, Weaver and Zakharov
%
% h_init is the initial value of h (should be a power of two)
% Mb is the number of bits used to code x
% Nu is the maximal number of successfull iterations.
%
% Trajectory variable is computed only for algorithm characterization.
%
% Francois Auger, Luo Zhen, april-july 2010
if (a==0)
 error('a should not be equal to zero');
elseif (a<0)
 a=-a; b=-b; % change both signs
end;

k=0; m=1; x=0; h=h_init; 
Dx=-2*b; a_times_h=a*h_init; % left or right shifts

if (nargout>=3), 
 Trajectory(:,1)=[x;a*x^2-2*b*x]; 
end;

while (m<=Mb)&(k<=Nu),
 while (abs(Dx)> a_times_h)
  if (Dx < 0.0)
   x=x+h; Dx=Dx + 2 * a_times_h; % left shift
  else
   x=x-h; Dx=Dx - 2 * a_times_h; % left shift
  end;

  k=k+1; 
  if (nargout>=3),
   Trajectory(:,k+1)=[x;a*x^2-2*b*x]; 
  end;

 end; % end while (abs(Dx)> a_times_h)
 m=m+1; h=h/2; a_times_h = a_times_h/2; % right shifts
end; % end while (m<=Mb)&(k<=Nu)
