function [logx,k,Trajectory]=dcd_log(x,Mb,Nu)
% [logx,k,Trajectory]=dcd_log(x,Mb,Nu)
% multiplication-free algorithm to compute a natural logarithm.
% Mb is the number of bits used to code x
% Nu is the maximal number of successfull iterations.
%
% To generate the table of logarithms, you can run the code below 
% in the command window and cut and paste the results in your code.
% Mb=20; 
% Base=exp(1);  % Natural logarithm in this case
% for i=0:Mb-1,
%  fprintf('LogTable(%2i)=%20.12e ;\n', i+1, log(1+2^(-i))/log(Base)); 
% end;

% Francois Auger,  november 2010
% This table is stored in memory
LogTable( 1)= 6.931471805599e-001 ;
LogTable( 2)= 4.054651081082e-001 ;
LogTable( 3)= 2.231435513142e-001 ;
LogTable( 4)= 1.177830356564e-001 ;
LogTable( 5)= 6.062462181643e-002 ;
LogTable( 6)= 3.077165866675e-002 ;
LogTable( 7)= 1.550418653597e-002 ;
LogTable( 8)= 7.782140442055e-003 ;
LogTable( 9)= 3.898640415657e-003 ;
LogTable(10)= 1.951220131262e-003 ;
LogTable(11)= 9.760859730555e-004 ;
LogTable(12)= 4.881620795014e-004 ;
LogTable(13)= 2.441108275274e-004 ;
LogTable(14)= 1.220628625257e-004 ;
LogTable(15)= 6.103329368064e-005 ;
LogTable(16)= 3.051711247319e-005 ;
LogTable(17)= 1.525867264836e-005 ;
LogTable(18)= 7.629365427568e-006 ;
LogTable(19)= 3.814689989686e-006 ;
LogTable(20)= 1.907346813825e-006 ;
LogTable(21)= 9.536738616592e-007 ;
LogTable(22)= 4.768370445163e-007 ;
LogTable(23)= 2.384185506799e-007 ;
LogTable(24)= 1.192092824454e-007 ;
LogTable(25)= 5.960464299903e-008 ;
LogTable(26)= 2.980232194361e-008 ;
LogTable(27)= 1.490116108283e-008 ;
LogTable(28)= 7.450580569168e-009 ;
LogTable(29)= 3.725290291523e-009 ;
LogTable(30)= 1.862645147496e-009 ;
LogTable(31)= 9.313225741818e-010 ;
LogTable(32)= 4.656612871993e-010 ;
LogTable(33)= 2.328306436268e-010 ;
LogTable(34)= 1.164153218202e-010 ;
LogTable(35)= 5.820766091177e-011 ;
LogTable(36)= 2.910383045631e-011 ;
LogTable(37)= 1.455191522826e-011 ;
LogTable(38)= 7.275957614157e-012 ;
LogTable(39)= 3.637978807085e-012 ;
LogTable(40)= 1.818989403544e-012 ;

if (x<=0), 
    error('x should be strictly positive'); 
end;

k=0; logx=0.0; 
if (nargout>=3), 
    Trajectory(:,k+1)=[x;logx;0]; 
end;

while (x>1.0)&&(k<Nu),
     k=k+1; x=x/2.0; logx=logx+LogTable(1);
     if (nargout>=3), 
         Trajectory(:,k+1)=[x;logx;0]; 
     end;
end;

while (x<=0.5)&&(k<Nu),
     k=k+1; x=x*2.0; 
     logx=logx-LogTable(1);
     if (nargout>=3), 
         Trajectory(:,k+1)=[x;logx;0]; 
     end;
end;

h=0.5; i=2;
while (i<Mb)&&(k<Nu),
     xNext=x+x*h; % one addition and one right shift
     if (xNext<1.0)&&(k<Nu),
          x=xNext; k=k+1; 
          logx=logx-LogTable(i); 
          xNext=x+x*h;
              if (nargout>=3), 
                  Trajectory(:,k+1)=[x;logx;i-1]; 
              end;
     end;
     h=h/2.0; i=i+1; % one right shift
end;
