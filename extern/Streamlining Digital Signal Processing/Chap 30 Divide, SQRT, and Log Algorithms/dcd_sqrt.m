
function [xn,k,Trajectory]=dcd_sqrt(c,h_init,Mb,Nu)
% [x,k,Trajectory]=dcd_sqrt(c,h_init,Mb,Nu)
% multiplication-free square root of c
% inspired from the algorithm proposed by Liu, Weaver and Zakharov
%
%
% h_init is the initial value of h (and should be a power of two)
% Mb is the number of bits used to code x
% Nu is the maximal number of successfull iterations.
%
% Trajectory variable is computed only for algorithm characterization.
%
% Francois Auger,  august 2010
if (c<=0), error('c should be strictly positive'); end;

m=1; xn=0; cn=c; h=h_init; 
hdiv2=h/2; htimes2=h*2; hsquare=h*h;  % left or right shift

k=0; if (nargout>=3), Trajectory(:,1)=[xn;cn]; end;

while (m<=Mb)&(k<=Nu),

 while (cn>=hsquare+xn*htimes2)||((cn+xn*htimes2>=hsquare)&&(xn<hdiv2))

  if (cn>=hsquare+xn*htimes2)         % left or right shift
   cn=cn-hsquare-xn*htimes2; xn=xn+h; % left or right shift
  else
   cn=cn-hsquare+xn*htimes2; xn=xn-h; % left or right shift
  end; % end if (cn>hsquare+xn*htimes2)

  k=k+1; if (nargout>=3), Trajectory(:,k+1)=[xn;cn]; end;

 end; % end while (cn>=hsquare+xn*htimes2)|((cn+xn*htimes2>=hsquare)&(xn<hdiv2))

 h=h/2; hdiv2=hdiv2/2; htimes2=htimes2/2; hsquare=hsquare/4; m=m+1; % right shift
end; % end while (m<=Mb)&(k<=Nu)
