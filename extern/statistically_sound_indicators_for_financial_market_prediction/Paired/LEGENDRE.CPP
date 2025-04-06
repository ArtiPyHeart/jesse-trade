/******************************************************************************/
/*                                                                            */
/*  LEGENDRE - Compute coeficients of discrete Legendre polynomial            */
/*                                                                            */
/******************************************************************************/

#include <assert.h>
#include <conio.h>
#include <float.h>
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>


#include "classes.h"
#include "const.h"
#include "funcdefs.h"


/*
--------------------------------------------------------------------------------

   Compute first and second-order normalized orthogonal coefs
   for n data points

--------------------------------------------------------------------------------
*/

void legendre_2(int n, double *c1, double *c2) {
  int i;
  double sum, mean;

  /*
     Compute c1
  */

  sum = 0.0;
  for (i = 0; i < n; i++) {
    c1[i] = 2.0 * i / (n - 1.0) - 1.0;
    sum += c1[i] * c1[i];
  }

  sum = sqrt(sum);
  for (i = 0; i < n; i++)
    c1[i] /= sum;

  /*
     Compute c2
  */

  sum = 0.0;
  for (i = 0; i < n; i++) {
    c2[i] = c1[i] * c1[i];
    sum += c2[i];
  }

  mean = sum / n; // Center it and normalize to unit length

  sum = 0.0;
  for (i = 0; i < n; i++) {
    c2[i] -= mean;
    sum += c2[i] * c2[i];
  }

  sum = sqrt(sum);
  for (i = 0; i < n; i++)
    c2[i] /= sum;
}
