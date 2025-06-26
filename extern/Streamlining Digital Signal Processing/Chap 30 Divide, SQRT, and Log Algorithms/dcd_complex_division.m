function [xr,xi,k,Trajectory]=dcd_complex_division(ar,ai,br,bi,h_init,Mb,Nu)
% [xr,xi,k,Trajectory]=dcd_complex_division(ar,ai,br,bi,H,Mb,Nu)
% multiplication-free division of b=br+j*bi by a=ar+j*ai 
% using the algorithm proposed by Liu, Weaver and Zakharov
%
% h_init is the initial value of h (and should be a power of two)
% Mb is the number of bits used to code x
% Nu is the maximal number of successfull iterations.

% Francois Auger, Luo Zhen, april-july 2010
if (ar==0)&(ai==0)
 error('a should not be equal to zero');
end;

if (abs(ai)>abs(ar))
 temp=br; br=-bi; bi=temp; temp=ar; ar=-ai; ai=temp; % multiply a and b by j
end;

if (ar<0)
 br=-br; bi=-bi; ar=-ar; ai=-ai; % multiply a and b by -1
end;

k=0; m=1; xr=0; xi=0; bnr=br; bni=bi; h=h_init; 
ar_times_h=ar*h_init; ai_times_h=ai*h_init; % left or right shifts

if (nargout>=3), 
 Trajectory(:,1)=[xr;xi];
end;

while (m<=Mb)&(k<=Nu),

 while (abs(bnr)> ar_times_h)
  if (bnr < 0.0)
   xr=xr-h; bnr=bnr + ar_times_h; bni=bni + ai_times_h;
  else
   xr=xr+h; bnr=bnr - ar_times_h; bni=bni - ai_times_h;
  end;

  k=k+1; % fprintf('x= %12.8e + j %12.8e\n', xr, xi);
  if (nargout>=3),
   Trajectory(:,k+1)=[xr;xi]; 
  end;

 end; % end while (abs(bnr)> ar_times_h)

 while (abs(bni)> ar_times_h)
  if (bni < 0.0)
   xi=xi-h; bnr=bnr - ai_times_h; bni=bni + ar_times_h;
  else
   xi=xi+h; bnr=bnr + ai_times_h; bni=bni - ar_times_h;
  end;

  k=k+1; % fprintf('x= %12.8e + j %12.8e\n', xr, xi);
  if (nargout>=3),
   Trajectory(:,k+1)=[xr;xi]; 
  end;

 end; % end while (abs(bni)> ar_times_h)

 m=m+1; 
 h=h/2; ar_times_h = ar_times_h/2; ai_times_h = ai_times_h/2; % right shifts
end; % end while (m<=Mb)&(k<=Nu)


