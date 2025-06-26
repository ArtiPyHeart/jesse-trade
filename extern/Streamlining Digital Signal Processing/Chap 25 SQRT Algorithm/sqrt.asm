/* SQRT.asm
	CONSTANTS	
 Prog	ADDRESSES	DEF: Sqrt,testLoWord,takeABS,OKToSqrt,doneSqrt
 Data	ADDRESSES	DEF: multx2,multx,multconst,AccelLookUpTable[12]
 Simulation Data ADDRESSES	DEF: beta,sqrt_hi, sqrt_lo,input_hi

 This code simulated with VisualDSP++3.5 218x tools

 This routine is NOT C callable.  It is intended for assembly calls.
 It wouldn't take much to change it to be C or C++ compliant.
 There is a wrapper around the routine for simulation.  The
 original routine uses a beta look up table.  This code does this
 or uses a linear or quadratic function of x to determine the
 beta factor.  Choose 1 method, Make sure the desired method is not
 commented out. Comment out the 2 that are not being used.

 Take the SQUARE ROOT of a Q31 number stored in the SR1 and SR0.
 An ABS will be taken if the input is negative.

 Good for all numbers >= 8000 0000h and <= 7fff ffff.

 SR1 and SR0 will contain the Q31 root.
 The result will have about 15 accurate bits.
 
 This subroutine uses a recursive non-linear filter to calculate the SQR.
 The filter will generate the Q31 root in 2 iterations with 15 accurate
 normalized mantissa bits. The algorithm used was published in the
 "IEEE Transactions on Signal Processing" July 1992 Volume 40 Number 7.
 The article name is "A new DSP-Oriented Algorithm for Calculation of the
 Square Root Using a Nonlinear Digital Filter" by N. Mikami, M. Kobayashi,
 Y. Yokoyama. Please see the article for a complete description of this
 algorithm.

	An initial guess is required. the formula to generate this is:

	Root(0) = 0.666667 * x + 0.354167

	where x is the number of which to take the square root.
	The equation of the recursive non-linear filter is:

	Root(n+1) = Beta * (x - Root(n)^2) + Root(n)   n: 0 to 1

	where Beta is an acceleration factor that depends on
	the Input signal, x. Beta is obtained from a look-up table. Root(n)
	starts at Root(0).

 The above formula only works well for numbers >= 0.25 and less than 1.
 This corresponds to Q31 values >= 02000 0000h and <= 07fff ffffh. Therefore
 x must be normalized to this range and the root must unnormalized
 after it is calculated. X is normalized by factors of 4 because
 de-normalizing for 4 is easy. ie. SQR(4) = 2 or SQR(1/4) = 1/2.

 Code written by Mark Allie, Oct. 2004.  Code to simulate 3 routines plus the routines.
*/

.global sqrt;

.section/data data1;
.var	multx2 = 0x61aa;
.var	multx = 0x6467;
.var	multconst = 0xabe7;
.var	beta, sqrt_hi, sqrt_lo, input_hi;
.var	accelLookupTable[12] = 0x7b20,0x6b90,0x6430,0x5e10,0x5880,
				 0x53c0,0x4fa0,0x4c30,0x4970,0x4730,0x4210,0x4060;
.section/code IVreset;
	jump start;
.section/code IVirq2;
.section/code IVirql1;
.section/code IVirql0;
.section/code IVsport0xmit;
.section/code IVsport0recv;
.section/code IVirqe;
.section/code IVbdma;
.section/code IVirq1;
.section/code IVirq0;
.section/code IVtimer;
.section/code IVpwrdwn;
.section/code program;
start:
	dis m_mode;			// Use Q31 arithmetic mode (auto left shift multplier results by
	l0  = 0x0000;		// Circular buffer indirect addressing off
	m0  = 0x0000;		// Indirect addressing modifier m0 set to 0
	m1  = 0x0001;		// Indirect addressing modifier m1 set to 1
	ay1 = 0x0020;		// Initialize counter to 0.25 * 32768  
						// initialize ay1 to a lower number to test normalization
testSQRT:
	sr1 = ay1;			// Set input to routine to counter value
	sr0 = 0x0000;
	call sqrt;			// Get the square root
	ar = ay1 + 1;
	ay1 = ar;			// increment the counter
	ar = ar - 0x8000;	// Once counter goes from 0x7fff to 0x8000 we are done
	ar = pass ar;
	if ne jump testSQRT;	

	nop;
	nop;
	nop;				 // Set a break point in here to stop sim when all number are tested
	nop;
	nop;
	jump start;
/* SQRT 
	Make sure the input x is greater than 0 before taking the square root.
	If it is less than 0 then make positive by multiplying by -1.  If 0 return 
	0 without performing the function. This testing does not need to be done
	if the user can guarantee that x is greater than 0 before calling the routine.
*/	
sqrt:
	mr0 = sr0;
	mr1 = sr1;
	my0 = 0x8000;
	ar = pass sr1;			// Make sure x is > 0
	if GT jump oKToSqrt;	// If > 0 get on with it already.
	if EQ jump testLoWord;	// Test 16 msb's of x
	if LT jump takeABS;		// If less than 0 mult by -1
	
testLoWord:
	ar = pass sr0;			// Test 16 lsb's of x
	if NE jump oKToSqrt;	// x was small take the square root
	jump doneSqrt;			// x was zero go home
takeABS:
	mr = sr0 * my0 (US);	// -1 * 16 lsbits of x
	mr0 = mr1;				// shift result right 16 bits
	mr1 = mr2;
	mr = mr + sr1 * my0 (SS);	// add -1 * 16 msbits of x This number could be 0x00 8000 0000
	if mv sat mr;			// if number was 0x00 8000 0000 make 0x00 7fff ffff

/*	Normalize 32 bit x. 10 cycles.
		Find the exponent of the 32 bit number ( It is always negative). Force the normalization exponent to be even
		and <= the original exponent.  There are many ways to do this.  The >>1  <<1 on a
		positive number seemed quicker than comparing methods even with the added abs and negate.
		Note >>1 <<1 on a negative number makes it even and greater in magnitude.
*/
oKToSqrt:	
	se = exp mr1 (HI);		// 32 bit exponent detect.
	se = exp mr0 (LO);
	ar = se;				// always negative
	ar = ABS ar;			// make exp pos. 
	sr = ashift ar by -1 (LO);	// + exp / 2 
	ar = - sr0;			// - exp / 2 make negative again for de-normalization
	sr = ashift AR by 1 (LO);	// even number of shifts < = exp
	se = sr0;					// se contains even exp for normalizing shift.
	sr = norm mr1 (HI);
	sr = sr or norm mr0 (LO), se = ar;  // sr contains x, set se now = exp / 2 to denorm sqrt later
					
/*	Beta Look up table root initialization.  8 cycles
		Right Shift normalized x, which will be 0x2000 00000 to 0x7fff ffff, by 11 bits.
		Keep the top 5 bits. 1 sign, always 0, and 4 significant mantissa bits.
		The significant nibble will be from 4 to 15.  Subtract 4 so nibble is between 0 and 11.
		Ad the table start address and store pointer in i0.  Save sr to ax and restore after 
		determining the address, since sr0 and sr1 hold the x which is used later.  Normally
		ax would be referenced below instead of sr and beta would be read directly into my1
		but this code can be quickly changed to implement 3 versions and this look up table 
		takes extra cycle hits for that convenience.

	ax0 = sr0;					// save x lo
	ax1 = sr1;					// save x hi
	sr = ashift sr1 by -11 (HI);// 5 msb are kept in SR1.  MSB = 0 because pos. number
	ay0 = accelLookupTable - 4;	// 4 to 15 becomes 0 to 11. Smallest possible number
								// is 4 because of normalization.
	ar = sr1 + ay0, mx1 = ax1;	// ar Beta LUT address. mx1 = x HI.
	i0 = ar;					// Index register set to beta address.
	mr1 = dm(i0,m0);			// mr1 = beta
	sr0 = ax0;					// Reload SR.
	sr1 = ax1;
*/
/*	Linear equation of x root initialization. 5 cycles.
		The equation is -0.61951*x + 1.0688.  The multiplier 1.0688 is divided by 2 for 
		convenient implementation.  The results using this multiplier are accumulated
		twice. The Minus 1 or 0x8000 is used. my0 is already set to 0x8000 above.
		This section of code could use constants stored in data memory or load the 
		constants into the m registers directly.  It takes the same number of cycles.
		I chose the direct method so I would not have to change the values stored in
		multx2,... every time I switched from linear to quadratic root initiation.
		Here is the data mem version of code.

			i0 = multx
			my1 = dm(i0,m1)
			mr = sr1 * my1 (SS), mx0 = dm(i0,m0);
			mr = mr + mx0 * my0;
			mr = ar + mx0 * my0; 
*/
	mx0 = 0xbb99;				// -0.5344
	my1 = 0xb0b4;				// -0.61951
	mr = mx0 * my0 (SS);		// -0.5344* (-1)
	mr = mr + mx0 * my0 (SS);	// 0.5344 + 0.5344 = 1.0688
	mr = mr + sr1 * my1 (SS);	// -0.61951 * x + 1.0688. Never >= 1	


/*	Quadratic equation of x root initialization. 8 cycles
		The equation is 0.763 * x^2  - 1.5688 * x + 1.314.  The constants > 1 and < 2
		are stored as constant/2 and implemented by accumulating the result twice. Minus
		1 or 0x8000 is used. my0 is already set to 0x8000 above.


	i0 = multx2;
	my1 = dm(i0,m1);			// 0.763 
	mr = sr1 * sr1 (SS);		// x^2
	mr = mr1 * my1 (SS),my1=dm(i0,m1);	// 0.763 * x^2, my1 = 0.7844
								
	mr = mr - sr1 * my1 (SS);	// 0.763*x^2 - 0.7844*x
	mr = mr - sr1 * my1 (SS),mx0=dm(i0,m0);	// 0.763*x^2 - 1.5688*x, mx0 = -0.657
								
	mr = mr + mx0 * my0 (SS);	// 0.763*x^2 - 1.5688*x + 0.657
	mr = mr + mx0 * my0 (SS);	// 0.763*x^2 - 1.5688*x + 1.314, Result is always < 1
*/								
/*	Iterations. 25 cycles including ret.  Does not include data memory writes at the end.
		This routine implements the non-linear iir filter described above.

		Root(0) = 0.666667 * x + 0.354167
		Root(n+1) = Beta * (x - Root(n)^2) + Root(n)   n: 0 to 1

		16 by 16 bit multiplies are used to save time.  Sometimes a small amount of
		error reduction is possible using 16 by 32 bit multiplies.  The mr register is 
		initialized to 0.324167 then the product of x by 2/3 is added.  This result,
		root(0), is rounded and	can be greater than 1 so it is saturated to 0x7fff ffff
		if needed. Load the mr register with x and subtract root(0)^2.  Multiply this
		result by beta and add root(0) which yields root(1). Round and saturate to
		0x7fff ffff. Repeat this whole process using root(1) which results in the 
		square root normalized.  De-normalize and we are done.

*/	// mr1 contains beta no matter which method from above is chosen.
	ar = mr1;				// store beta. 
	
	my1 = 0x5553;	   		// 2/3 * 2^15
						
	mr1 = 0x2d55;	   		// 0.354167 *2^31
	mr0 = 0x5821;	   		// Not really needed
	mr = mr + sr1 * my1 (SS), my1 = ar;	// 2/3 * x HI + .354167
										// my1 = ar = Beta Factor 
	mr = mr (RND);			// Round
	if mv sat mr;			// limit to 0x7fff ffff 
	mx1 = mr1;				// root(0)
	
	mr1 = sr1;				// mr1 = x HI
	mr0 = sr0;				// mr x HI + LO
	mr = mr - mx1 * mx1 (SS);	// mr = x - root(0)^2
										
	mr = mr1 * my1 (SS);	// root(1) = Beta * ( x - root(0)^2 )
	mr = mr - mx1 * my0 (SS);	// root(1) = Beta * (x - root(0)^2) - (-1)*root(0)
	mr = mr (RND);			// round
	if mv sat mr;			// limit to 0x7fff ffff
	mx1 = mr1;				// 16 bit rounded limited root(1)
	
	mr1 = sr1;				// mr1 = x HI
	mr0 = sr0;				// mr = x HI + x LO
	mr = mr - mx1 * mx1 (SS);	// x - root(1)^2
	mr = mr1 * my1 (SS);	// root(2) = Beta * ( x - root(1)^2 )
	mr = mr - mx1 * my0 (SS);	// root(2) = Beta * (x - root(1)^2) - (-1)*root(1)
	mr = mr (RND);			// round
	if mv sat mr;			// limit to 0x7fff ffff
	sr = ashift mr1 (HI);	// de-normalize
	sr = sr or lshift mr0 (LO);	// de-normalize  32bit result
doneSqrt:
	dm(sqrt_hi) = sr1;		// Save the input,output and beta in data mem.
	dm(sqrt_lo) = sr0;		// Use streaming in the simulator to save results.
	dm(input_hi) = ay1;
	dm(beta) = ar;
	rts;
	
sqrt.END:
