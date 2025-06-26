
function [x,k,Trajectory]= dcd_real_matrix(A,b,h_init,Mb,Nu)
% Dichotomous Coordinate Descent algorithm applied to the solution of a linear system
%
% [x,k,Trajectory]= dcd_real_matrix(A,b,h_init,Mb,Nu)
% solves the linear equation Ax=b, where R is a symmetric matrix,
% using the algorithm proposed by Zakharov and T.C.Tozer
%
% h_init is the initial value of h (and should be a power of two)
% Mb is the number of bits used for the binary representation of the elements of vector x
% Nu is the maximal number of successfull iterations.
%
% Trajectory variable is computed only for algorithm characterization.
%
% Francois Auger, Luo Zhen, May-july 2010

m=1; [Nr,Nc]=size(A); x=zeros(Nr,1); Dx=-2*b; h=h_init; 

k=0; if (nargout>=3), Trajectory(:,1)=[x;x'*A*x-2*b'*x]; end;

while (m<=Mb)&(k<=Nu),
 Flaga=1;
 while (Flaga==1)
  Flaga=0;
  for i=1:Nr,
   if (abs(Dx(i)) >= h*A(i,i))
    if (Dx(i)>=0)
     x(i)=x(i)-h; Dx=Dx-2*h*A(:,i);
    else
     x(i)=x(i)+h; Dx=Dx+2*h*A(:,i);
    end; % if (Dx(i)>0)
    Flaga=1;

    k=k+1; if (nargout>=3), Trajectory(:,k+1)=[x;x'*A*x-2*b'*x]; end;
   end; % end if (abs(Dx(i)) > h*A(i,i))
  end; % for i=1:Nr
 end; % while (Flaga==1)
 h=h/2; m=m+1;
end;
