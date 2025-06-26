

// Scallop Loss Routine by Clay S. Turner - 20 Sept 2010 V1.0
//
// This program is intended to be an educational tool for
// showing how to simply compute the scallop loss of various 
// windows via DFTs and convolution.
// 
// This function finds the scallop loss associated with common 
// windows (rectangular, von Hann, Hamming) and the 3 term
// Salvatore Flat Top window (SFT3F). Additionally a couple of approximations
// to the Salvatore window are measured.
//
// SFT3F refers to Salvatore Flat Top 3 term Fastest falloff
//
// The scallop loss is measured by generating a set of sinusoids
// on a fine grid of frequencies and for each frequency, a set of 5 DFTs for
// the nearest bins are calculated. Then using the multiplication - convolution
// theorem for DFTs, the equivalent DFT of the nearest bin with windowed data is calculated.
// Then the magnitude of that result is found. The results for each frequency 
// and window type is output in Excel .csv file where each column is for a 
// different window and each row is for a different frequency.
// Excel's graphing capabilites may be easily used to examine the results
// in various ways.
//
// To simplify the code, a rudimentary class is defined for handling complex numbers.
// This class is not intended to be complete but rather it is to have just enough
// capability to simplify the computations.
//

#include "stdafx.h"
#include <math.h>

const double pi2=6.2831853071796;	// 2 * Archemedian constant
const int N=1024;					// # of data points in DFT
const int fc=N/4;					// bin number with respect the computations are centered about
const int Nfreqs=1000;				// number of frequencies used in analysis - affects output file size

// defines the interface for the "complex" class

class CPX {
private:
	double x;						// the real component
	double y;						// the imaginary component
public:
//	CPX(void);
	CPX(double re=0,double im=0);
	CPX operator+(const CPX a) const;					// returns sum of complex
	CPX operator-(const CPX a) const;					// returns difference of complex
	friend CPX operator*(const double a,const CPX b);	// returns product of scalar times complex
	friend CPX operator*(const CPX b,const double a);	// returns product of complex times scalar
	friend CPX operator/(const CPX b,const double a);	// returns quotient of complex divided by scalar
	double mag();										// returns magnitude of complex number
	};

// defines the implementation for the "complex" class

//CPX::CPX(void)		// default constructor
//{
//x=0;
//y=0;
//}

CPX::CPX(double re, double im)	// initialized constructor
{
x=re;
y=im;
}


CPX CPX::operator+(const CPX b) const	// CPX addition
{
CPX c(x+b.x,y+b.y);
return c;
}

CPX CPX::operator-(const CPX b) const	// CPX subtraction
{
CPX c(x-b.x,y-b.y);
return c;
}


CPX operator*(const double a,const CPX b)	// multiply CPX by scalar

{
CPX c(b.x * a,b.y * a);
return c;
}


CPX operator*(const CPX b,const double a)	// multiply CPX by scalar
{
CPX c(b.x * a,b.y * a);
return c;
}

CPX operator/(const CPX b,const double a)	// divide CPX by scalar
{
CPX c(b.x / a,b.y / a);
return c;
}

double CPX::mag()		// returns magnitude of complex number
{
return sqrt(x*x+y*y);
}


// perform DFT at freq 'k' on 'data'
//
// input:
//       data            vector ( length N) of real valued data to comput the DFT of
//       k               bin number used for DFT analysis
//
// output:
//       Complex valued object containing real and imaginarg components of DFT normed by 2/N
//
CPX DFT(double data[], int k)
{
double x,y,w,*ptr;
int n;
CPX c;

w=pi2*k/N;	// scale factor for inside of DFT computation
for (n=0,x=0.0,y=0.0,ptr=data;n<N;n++,ptr++) {
	x+= *ptr * cos(w*n);
	y-= *ptr * sin(w*n);
	}
c=CPX(x,y)*(2.0/N);	// normalize result so "c" has amplitude equal to sinusoid's amplitude
return c;
}


// Scallop Loss calculations
// Instead of windowing the time domain data, the windows get "applied" by performing
// convolution in the frequency domain. All of the windows involved in this anaysis
// may be written in terms of low order trigonometric polynomials. Hence their synthesis
// via convolution only involves a few DFT terms.
//
// inputs:
//        name			name of output file for data
//        lo            starting bin frequency for analysis
//        hi            ending bin frequency for analysis
//        Npts          Number (actually one more) of frequecies anaysis is performed over
//
// output:
//        .csv (comma separated values) file readable in Microsoft's Excel
//
bool ScallopLoss(char name[],double lo, double hi, int Npts)
{
FILE *fpout;
double df, f, m, data[N];
CPX a,b,c,d,e,g,p,q,r;
int n,bin;

if (NULL==(fpout=fopen(name,"w"))) {
	printf("Error: Unable to open file <%s> for writing.\n",name);
	return false;
	}

// write out column headers for .csv file
fprintf(fpout,"Freq.,Rectangular,von Hann,Hamming,SFT3F (full),SFT3F (4 bit),SFT3F (8 bit)\n");

// now get ready to step thru all frequencies and at each freq, 
// calc. scallop loss for each window type.

df=(hi-lo)/Npts;						// freq step
for (f=lo;f<=hi;f+=df) {

	fprintf(fpout,"%.15f,",f-fc);		// output freq relative to center freq

	for (n=0;n<N;n++)
		data[n]=cos(pi2*f*n/N);			// create data vector of test sinusoid at frequency 'f'

	bin=(int)(0.5+f);					// reference bin number is one nearest test frequency

	a=DFT(data,bin-2);					// X[bin-2]
	b=DFT(data,bin-1);					// X[bin-1]
	c=DFT(data,bin);					// X[bin]
	d=DFT(data,bin+1);					// X[bin+1]
	e=DFT(data,bin+2);					// X[bin+2]

	p=b+d;								// X[bin-1] + X[bin+1]
	q=a+e;								// X[bin-2] + X[bin+2]
	r=q-p;								// X[bin-2] + X[bin+2] - X[bin-1] - X[bin-2]

	fprintf(fpout,"%.15f,",c.mag());	// output Rect window scallop loss

	g=c-0.5*p;							// von Hann window via convolution
	fprintf(fpout,"%.15f,",g.mag());	// output scallop loss

	g=c-0.4259259*p;					// Hamming window via convolution
	fprintf(fpout,"%.15f,",g.mag());	// output scallop loss 

	g=c-0.94247*p+0.44247*q;			// Salvatore 3 Term Flat Top window via convolution (full precision)
	fprintf(fpout,"%.15f,",g.mag());	// output Flat Top window scallop loss

	g=c-p+(q-r/8.0)/2.0;				// Lyons' 4 bit approximation to Salvatore Flat Top window via convolution
	fprintf(fpout,"%.15f,",g.mag());	// output scallop loss

	g=c-p+(q+(r/16.0-r)/8.0)/2.0;		// Lyons' 8 bit approximation to Salvatore Flat Top window via convolution
	fprintf(fpout,"%.15f",g.mag());		// output scallop loss

	fprintf(fpout,"\n");				// end of row of data
	}
fclose(fpout);
return true;
}


int main(int argc, char* argv[])
{
char name[]="Chart.csv";

printf("\n\nScallop Loss Computation Program (V1.0) 20 Sept 2010 by Clay S. Turner\n\n");
if (true==ScallopLoss(name,fc,fc+1,Nfreqs)) 
	printf("Scallop Loss Computation Complete; <%s> written\n",name);
else 
	printf("Error: Scallop Loss Computation did not complete!\n");

return 0;
}

