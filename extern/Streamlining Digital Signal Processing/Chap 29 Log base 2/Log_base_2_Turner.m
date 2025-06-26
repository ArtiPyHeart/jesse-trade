
%  Filename: Log_base_2_Turner.m

%  Models "Binary Logarithm Algorithm"
%  developed by Clay S. Turner.
%
%     Coded by: Richard Lyons [June 2010]

clear, clc

% Desired number of bits in result "Y"
    Num_Y_Bits = 3; 

% Define value of input "X", and compute 'true' log base 2 of "X"
    X = 100.0045;
    
    disp(' '), disp(' ')
    disp(['X = ', num2str(X)])
    Y_True = log2(X);

% Compute 'characteristic' of "Y"
    Y_char = 0; % Init the caharacteristis
    if X >= 2
        while X >= 2
            X = X/2;
            Y_char = Y_char + 1;
        end
    elseif X < 1
        while X < 1
            X = 2*X;
            Y_char = Y_char -1;
        end
    end

% Compute 'mantissa' of "Y"
	Temp = X*X;  % Init a temp register
	
	for Loop = 1:Num_Y_Bits
        if Temp >= 1.99999999999999
            %disp('Temp was >= 2, set a bit')
            Y_Binary(Loop) = 1;
            Temp = Temp/2;
            Temp = Temp*Temp;
        else
            %disp('Temp was < 2, DO NOT set a bit')
            Y_Binary(Loop) = 0;
            Temp = Temp*Temp;
        end
	end
	
	Y_Decimal = 0;  % Init "Y_Decimal"
	
	for Loop = 1:Num_Y_Bits
        Y_Decimal = Y_Decimal + Y_Binary(Loop)/(2^Loop);
	end

% Compute 'characteristic' of "Y"
    Y_Decimal = Y_char + Y_Decimal;

% Evaluate the compute log base2 of "X"
    disp(['True "Y" = ', num2str(Y_True)])
    disp(['Estimated "Y" = ', num2str(Y_Decimal)])

    Error = Y_True - Y_Decimal;
    disp(['Error = ', num2str(Error)])
    Percent_Error = 100*(Error/Y_True);
    disp(['Percent Error = ', num2str(Percent_Error)])


