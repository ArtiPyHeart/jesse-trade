/******************************************************************************/
/*                                                                            */
/*  PAIRED - Paired-market indicators                                         */
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


#define MKTBUF 2048 /* Alloc for market info in chunks of this many records */
                    /* This is not critical and can be any reasonable vlaue */

/*
   These are defined in MEM64.CPP
   This code is needed only if MALLOC maps to memalloc et cetera.
   If MALLOC maps to malloc (etc) then this is not needed.
   This is controlled by the MEMDEBUG flag in CONST.H.
*/

#if MEMDEBUG
extern int mem_keep_log;     // Keep a log file?
extern char mem_file_name[]; // Log file name
extern __int64 mem_max_used; // Maximum memory ever in use
#endif

/*
--------------------------------------------------------------------------------

   Local routines get parameters from user's line

--------------------------------------------------------------------------------
*/

void get_1_param(char *lptr, double lower1, double upper1, double *param1) {
  while (*lptr == ' ' || *lptr == '\t')
    ++lptr;
  *param1 = atof(lptr);
  if (*param1 < lower1)
    *param1 = lower1;
  if (*param1 > upper1)
    *param1 = upper1;
}

void get_2_params(char *lptr, double lower1, double upper1, double lower2,
                  double upper2, double *param1, double *param2) {
  while (*lptr == ' ' || *lptr == '\t')
    ++lptr;
  *param1 = atof(lptr);
  if (*param1 < lower1)
    *param1 = lower1;
  if (*param1 > upper1)
    *param1 = upper1;
  while (isgraph(*lptr))
    ++lptr;
  while (*lptr == ' ' || *lptr == '\t')
    ++lptr;
  *param2 = atof(lptr);
  if (*param2 < lower2)
    *param2 = lower2;
  if (*param2 > upper2)
    *param2 = upper2;
}

void get_3_params(char *lptr, double lower1, double upper1, double lower2,
                  double upper2, double lower3, double upper3, double *param1,
                  double *param2, double *param3) {
  while (*lptr == ' ' || *lptr == '\t')
    ++lptr;
  *param1 = atof(lptr);
  if (*param1 < lower1)
    *param1 = lower1;
  if (*param1 > upper1)
    *param1 = upper1;
  while (isgraph(*lptr))
    ++lptr;
  while (*lptr == ' ' || *lptr == '\t')
    ++lptr;
  *param2 = atof(lptr);
  if (*param2 < lower2)
    *param2 = lower2;
  if (*param2 > upper2)
    *param2 = upper2;
  while (isgraph(*lptr))
    ++lptr;
  while (*lptr == ' ' || *lptr == '\t')
    ++lptr;
  *param3 = atof(lptr);
  if (*param3 < lower3)
    *param3 = lower3;
  if (*param3 > upper3)
    *param3 = upper3;
}

void get_4_params(char *lptr, double lower1, double upper1, double lower2,
                  double upper2, double lower3, double upper3, double lower4,
                  double upper4, double *param1, double *param2, double *param3,
                  double *param4) {
  while (*lptr == ' ' || *lptr == '\t')
    ++lptr;
  *param1 = atof(lptr);
  if (*param1 < lower1)
    *param1 = lower1;
  if (*param1 > upper1)
    *param1 = upper1;
  while (isgraph(*lptr))
    ++lptr;
  while (*lptr == ' ' || *lptr == '\t')
    ++lptr;
  *param2 = atof(lptr);
  if (*param2 < lower2)
    *param2 = lower2;
  if (*param2 > upper2)
    *param2 = upper2;
  while (isgraph(*lptr))
    ++lptr;
  while (*lptr == ' ' || *lptr == '\t')
    ++lptr;
  *param3 = atof(lptr);
  if (*param3 < lower3)
    *param3 = lower3;
  if (*param3 > upper3)
    *param3 = upper3;
  while (isgraph(*lptr))
    ++lptr;
  while (*lptr == ' ' || *lptr == '\t')
    ++lptr;
  *param4 = atof(lptr);
  if (*param4 < lower4)
    *param4 = lower4;
  if (*param4 > upper4)
    *param4 = upper4;
}

/*
-------------------------------------------------------------------------------

   Local routine to read market prices and volume if present

-------------------------------------------------------------------------------
*/

static int read_market(char *MarketName, int **date, double **open,
                       double **high, double **low, double **close,
                       double **volume, int *nprices) {
  int i, bufcnt;
  char line[256], *cptr;
  FILE *fp;

  if (fopen_s(&fp, MarketName, "rt")) {
    printf("\n\nCannot open market history file %s", MarketName);
    return ERROR_FILE;
  }

  *date = (int *)MALLOC(MKTBUF * sizeof(int));
  *open = (double *)MALLOC(MKTBUF * sizeof(double));
  *high = (double *)MALLOC(MKTBUF * sizeof(double));
  *low = (double *)MALLOC(MKTBUF * sizeof(double));
  *close = (double *)MALLOC(MKTBUF * sizeof(double));
  *volume = (double *)MALLOC(MKTBUF * sizeof(double));
  if (*date == NULL || *open == NULL || *high == NULL || *low == NULL ||
      *close == NULL || *volume == NULL) {
    printf("\n\nInsufficient memory reading market history file %s",
           MarketName);
    fclose(fp);
    if (*date != NULL)
      FREE(*date);
    if (*open != NULL)
      FREE(*open);
    if (*high != NULL)
      FREE(*high);
    if (*low != NULL)
      FREE(*low);
    if (*close != NULL)
      FREE(*close);
    if (*volume != NULL)
      FREE(*volume);
    return ERROR_INSUFFICIENT_MEMORY;
  }

  bufcnt = MKTBUF; // There are this many record slots available now

  printf("\nReading market file...");

  *nprices = 0; // Counts lines (prices) read

  for (;;) {

    if (feof(fp)                          // If end of file
        || (fgets(line, 256, fp) == NULL) // Or unable to read line
        || (strlen(line) < 2))            // Or empty line
      break;                              // We are done reading price history

    if (ferror(fp)) { // If an error reading file
      fclose(fp);     // Quit immediately
      printf("\nError reading line %d of file %s", *nprices + 1, MarketName);
      return ERROR_FILE;
    }

    if (!bufcnt) { // Allocate a new memory block if needed
      *date = (int *)REALLOC(*date, (*nprices + MKTBUF) * sizeof(int));
      *open = (double *)REALLOC(*open, (*nprices + MKTBUF) * sizeof(double));
      *high = (double *)REALLOC(*high, (*nprices + MKTBUF) * sizeof(double));
      *low = (double *)REALLOC(*low, (*nprices + MKTBUF) * sizeof(double));
      *close = (double *)REALLOC(*close, (*nprices + MKTBUF) * sizeof(double));
      *volume =
          (double *)REALLOC(*volume, (*nprices + MKTBUF) * sizeof(double));
      if (*date == NULL || *open == NULL || *high == NULL || *low == NULL ||
          *close == NULL || *volume == NULL) {
        fclose(fp);
        printf("\n\nInsufficient memory reading market history file %s  Press "
               "any key...",
               MarketName);
        if (*date != NULL)
          FREE(*date);
        if (*open != NULL)
          FREE(*open);
        if (*high != NULL)
          FREE(*high);
        if (*low != NULL)
          FREE(*low);
        if (*close != NULL)
          FREE(*close);
        if (*volume != NULL)
          FREE(*volume);
        return ERROR_INSUFFICIENT_MEMORY;
      } // If insufficient memory
      bufcnt = MKTBUF; // There are this many new record slots available now
    } // If allocating new block

    // Parse the date and do a crude sanity check

    for (i = 0; i < 8; i++) {
      if ((line[i] < '0') || (line[i] > '9')) {
        fclose(fp);
        printf("\nInvalid date reading line %d of file %s", *nprices + 1,
               MarketName);
        FREE(*date);
        FREE(*open);
        FREE(*high);
        FREE(*low);
        FREE(*close);
        FREE(*volume);
        return ERROR_SYNTAX;
      }
    }
    (*date)[*nprices] = atoi(line);

    // Parse the open

    cptr = line + 9; // Price is in this column or beyond
                     // (Next loop allows price to start past this)

    while (*cptr == ' ' || *cptr == '\t' || *cptr == ',') // Delimiters
      ++cptr; // Move up to the price

    (*open)[*nprices] = atof(cptr);

    while (*cptr && *cptr != ' ' && *cptr != ',' && *cptr != '\t')
      ++cptr; // Pass the price; stop at delimiter

    // Parse the high

    while (*cptr == ' ' || *cptr == '\t' || *cptr == ',') // Delimiters
      ++cptr; // Move up to the price

    (*high)[*nprices] = atof(cptr);

    while (*cptr && *cptr != ' ' && *cptr != ',' && *cptr != '\t')
      ++cptr; // Pass the price; stop at delimiter

    // Parse the low

    while (*cptr == ' ' || *cptr == '\t' || *cptr == ',') // Delimiters
      ++cptr; // Move up to the price

    (*low)[*nprices] = atof(cptr);

    while (*cptr && *cptr != ' ' && *cptr != ',' && *cptr != '\t')
      ++cptr; // Pass the price; stop at delimiter

    // Parse the close

    while (*cptr == ' ' || *cptr == '\t' || *cptr == ',') // Delimiters
      ++cptr; // Move up to the price

    (*close)[*nprices] = atof(cptr);

    while (*cptr && *cptr != ' ' && *cptr != ',' && *cptr != '\t')
      ++cptr; // Pass the price; stop at delimiter

    // Parse the volume

    while (*cptr == ' ' || *cptr == '\t' || *cptr == ',') // Delimiters
      ++cptr;                                             // Move up

    (*volume)[*nprices] = atof(cptr);

    if ((*low)[*nprices] > (*open)[*nprices] ||
        (*low)[*nprices] > (*close)[*nprices] ||
        (*high)[*nprices] < (*open)[*nprices] ||
        (*high)[*nprices] < (*close)[*nprices]) {
      fclose(fp);
      printf("\nInvalid open/high/low/close reading line %d of file %s",
             *nprices + 1, MarketName);
      FREE(*date);
      FREE(*open);
      FREE(*high);
      FREE(*low);
      FREE(*close);
      FREE(*volume);
      return ERROR_SYNTAX;
    }

    ++*nprices;
    --bufcnt; // One less slot remains

  } // For all lines

  fclose(fp);

  printf("\nMarket price history read; %d cases from %d to %d", *nprices,
         (*date)[0], (*date)[*nprices - 1]);
  return 0;
}

/*
--------------------------------------------------------------------------------

   Main routine

--------------------------------------------------------------------------------
*/

int main(int argc,    // Number of command line arguments (includes prog name)
         char *argv[] // Arguments (prog name is argv[0])
) {
  int i, k, line_num, icase, nvars, nprices, nprices1, nprices2, var, *date1,
      *date2;
  int n_done, first_date, last_date, ret_val, front_bad;
  int market_index_1, market_index_2, grand_index, max_date;
  double param1, param2, param3, param4;
  double *open1, *high1, *low1, *close1, *volume1;
  double *open2, *high2, *low2, *close2, *volume2;
  double *var_work, *vptr, *vars[MAX_VARS];
  double *work1, *work2, *work3, var_min, var_max, var_mean, var_iqr, var_ent;
  char line[256], MarketName1[4096], MarketName2[4096], ScriptName[4096], *mptr,
      *lptr;
  char user_name[MAX_NAME_LENGTH + 1];
  char var_names[MAX_VARS][MAX_NAME_LENGTH + 1];
  FILE *fp;
  SYSTEMTIME systime;

  date1 = NULL;
  open1 = high1 = low1 = close1 = volume1 = NULL;
  date2 = NULL;
  open2 = high2 = low2 = close2 = volume2 = NULL;
  var_work = work1 = work2 = work3 = NULL;
  for (i = 0; i < MAX_VARS; i++)
    vars[i] = NULL;
  nvars = 0;

  /*
     Process command line parameters
  */

#if 1
  if (argc != 4) {
    printf("\nUsage: PAIRED  Market1Name  Market2Name  ScriptName");
    printf("\n  Market1Name - name of first market file (YYYYMMDD Open High "
           "Low Close)");
    printf("\n  Market2Name - name of second (reference) market file");
    printf("\n  ScriptName - name of variable script file");
    exit(1);
  }

  strcpy_s(MarketName1, argv[1]);
  strcpy_s(MarketName2, argv[2]);
  strcpy_s(ScriptName, argv[3]);
#else
  strcpy_s(MarketName1,
           "E:\\MarketDataAssorted\\SP100\\IBM.TXT"); // For diagnostics only
  strcpy_s(MarketName2, "E:\\MarketDataAssorted\\INDEXES\\$OEX.TXT");
  strcpy_s(ScriptName, "VP.TXT");
#endif

  /*
     Memory checking stuff for MEM64.CPP safe memory allocation.
     This code is needed only if MALLOC maps to memalloc et cetera.
     If MALLOC maps to malloc (etc) then this is not needed.
     This is controlled by the MEMDEBUG flag in CONST.H.
  */

#if MEMDEBUG
  mem_max_used = 0;
  mem_keep_log = 1;
  _fullpath(mem_file_name, "MEM.LOG", 256);
  fp = fopen(mem_file_name, "wt");
  if (fp == NULL) {
    mem_file_name[0] = 0;
    mem_keep_log = 0;
  } else {
    GetLocalTime(&systime);
    sprintf(line, "PAIRED memory log %s  %d/%d/%02d  %d:%02d:%02d",
            mem_file_name, systime.wMonth, systime.wDay, systime.wYear % 100,
            systime.wHour, systime.wMinute, systime.wSecond);
    fprintf(fp, "%s", line);
    fclose(fp);
  }
#endif

  /*
  -------------------------------------------------------------------------------

     Read the two market files, then merge the records for identical dates

  -------------------------------------------------------------------------------
  */

  if (read_market(MarketName1, &date1, &open1, &high1, &low1, &close1, &volume1,
                  &nprices1)) {
    printf("\nFile error reading %s... aborting", MarketName1);
    goto FINISH;
  }

  if (read_market(MarketName2, &date2, &open2, &high2, &low2, &close2, &volume2,
                  &nprices2)) {
    printf("\nFile error reading %s... aborting", MarketName2);
    goto FINISH;
  }

  market_index_1 = market_index_2 = grand_index =
      0; // Index for each market and merged
  for (;;) {
    max_date = (date1[market_index_1] > date2[market_index_2])
                   ? date1[market_index_1]
                   : date2[market_index_2];

    while (market_index_1 < nprices1) { // Must not over-run a market!
      if (date1[market_index_1] >=
          max_date) // Advance mkt1 to (or past!) max_date
        break;
      ++market_index_1;
    }

    if (market_index_1 >= nprices1) // Done if we hit end of this market
      break;

    while (market_index_2 < nprices2) {
      if (date2[market_index_2] >=
          max_date) // Advance mkt2 to (or past!) max_date
        break;
      ++market_index_2;
    }

    if (market_index_2 >= nprices2) // Done if we hit end of this market
      break;

    if (date1[market_index_1] ==
        date2[market_index_2]) { // Both markets are at the same date
      date1[grand_index] = date1[market_index_1];
      date2[grand_index] =
          date2[market_index_2]; // We really need only one of these set
      open1[grand_index] = open1[market_index_1];
      open2[grand_index] = open2[market_index_2];
      high1[grand_index] = high1[market_index_1];
      high2[grand_index] = high2[market_index_2];
      low1[grand_index] = low1[market_index_1];
      low2[grand_index] = low2[market_index_2];
      close1[grand_index] = close1[market_index_1];
      close2[grand_index] = close2[market_index_2];
      volume1[grand_index] = volume1[market_index_1];
      volume2[grand_index] = volume2[market_index_2];
      ++market_index_1;
      ++market_index_2;
      ++grand_index;
    }

  } // Endless loop that merges all records

  nprices = grand_index;

  printf("\nFinal dataset has %d dates ranging from %d through %d", nprices,
         date1[0], date1[nprices - 1]);

  if (nprices == 0) {
    printf("\nAborting because there are no common dates");
    goto FINISH;
  }

  /*
  -------------------------------------------------------------------------------

     Read and process variable script file

  -------------------------------------------------------------------------------
  */

  if (fopen_s(&fp, ScriptName, "rt")) {
    printf("\n\nCannot open variable script file %s", ScriptName);
    goto FINISH;
  }

  var_work = (double *)MALLOC(nprices * sizeof(double));
  work1 = (double *)MALLOC(nprices * sizeof(double));
  work2 = (double *)MALLOC(nprices * sizeof(double));
  work3 = (double *)MALLOC(nprices * sizeof(double));
  if (var_work == NULL || work1 == NULL || work2 == NULL || work3 == NULL) {
    printf("\n\nInsufficient memory processing script file");
    fclose(fp);
    goto FINISH;
  }

  param1 = param2 = param3 = param4 = 0.0; // Not needed, shuts up LINT
  nvars = 0;                               // Counts variables
  front_bad = 0; // Keeps track of unitialized cases at start
  printf(
      "\n\nVarNum      Variable  N cases    First date   Last date          "
      "Mean       Minimum       Maximum    IQ Range     Rng/IQR  Rel Entropy");

  for (line_num = 1;; line_num++) {

    if (feof(fp)                          // If end of file
        || (fgets(line, 256, fp) == NULL) // Or unable to read line
        || (strlen(line) < 2))            // Or empty line
      break;                              // We are done reading price history

    if (ferror(fp)) { // If an error reading file
      fclose(fp);     // Quit immediately
      printf("\nError reading line %d of file %s", line_num, ScriptName);
      goto FINISH;
    }

    /*
       Change line to all upper case for uniformity, then delete any comments
    */

    _strupr(line);

    mptr = strchr(line, ';');
    if (mptr != NULL)
      *mptr = NULL;

    if (strlen(line) < 2) // Ignore blank lines
      continue;

    /*
       Parse the variable's user name
    */

    // Copy the user's name for this variable
    lptr = line;
    mptr = &user_name[0];
    while (*lptr == ' ' || *lptr == '\t') // Bypass leading blanks
      ++lptr;
    k = 0; // Will count name length
    while (*lptr && *lptr != ' ' && *lptr != '\t' && *lptr != ':') {
      ++k;
      if (k > MAX_NAME_LENGTH) {
        fclose(fp); // Quit immediately
        printf("\nUser name longer than %d characters in line %d",
               MAX_NAME_LENGTH, line_num);
        goto FINISH;
      }
      *mptr++ = *lptr++;
    }
    *mptr = 0;

    for (i = 0; i < nvars; i++) {
      if (!strcmp(user_name, var_names[i])) {
        fclose(fp); // Quit immediately
        printf("\nUser name %s duplicated in line %d", user_name, line_num);
        goto FINISH;
      }
    }
    strcpy(var_names[nvars], user_name);

    // Bypass colon and blanks to get to parser name of variable
    while (*lptr == ' ' || *lptr == '\t' || *lptr == ':')
      ++lptr;

    /*
       Determine which variable it is and get necessary parameters
    */

    if (!strncmp(lptr, "CORRELATION", 11)) {
      var = VAR_CORRELATION;
      lptr += 11;
      get_1_param(lptr, 2.0, nprices / 2, &param1);
    }

    else if (!strncmp(lptr, "DELTA CORRELATION", 17)) {
      var = VAR_DELTA_CORRELATION;
      lptr += 17;
      get_2_params(lptr, 2.0, nprices / 2, 1.0, nprices / 2, &param1, &param2);
    }

    else if (!strncmp(lptr, "DEVIATION", 9)) {
      var = VAR_DEVIATION;
      lptr += 9;
      get_2_params(lptr, 2.0, nprices / 2, 1.0, nprices / 2, &param1, &param2);
    }

    else if (!strncmp(lptr, "PURIFY", 6)) {
      var = VAR_PURIFY;
      lptr += 6;
      get_4_params(lptr, 3.0, nprices / 2, 0.0, nprices / 2, 0.0, nprices / 2,
                   0.0, nprices / 2, &param1, &param2, &param3, &param4);
      if (param2 <= 0.0 && param3 <= 0.0 && param4 <= 0.0) {
        fclose(fp); // Quit immediately
        printf("\n\nERROR... PURIFY cannot have all lookbacks zero");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "LOG PURIFY", 10)) {
      var = VAR_LOG_PURIFY;
      lptr += 10;
      get_4_params(lptr, 3.0, nprices / 2, 0.0, nprices / 2, 0.0, nprices / 2,
                   0.0, nprices / 2, &param1, &param2, &param3, &param4);
      if (param2 <= 0.0 && param3 <= 0.0 && param4 <= 0.0) {
        fclose(fp); // Quit immediately
        printf("\n\nERROR... PURIFY cannot have all lookbacks zero");
        goto FINISH;
      }
    }

    else if (!strncmp(lptr, "TREND DIFF", 10)) {
      var = VAR_TREND_DIFF;
      lptr += 10;
      get_2_params(lptr, 2.0, nprices / 2, 1.0, nprices / 2, &param1, &param2);
    }

    else if (!strncmp(lptr, "CMMA DIFF", 9)) {
      var = VAR_CMMA_DIFF;
      lptr += 9;
      get_2_params(lptr, 2.0, nprices / 2, 1.0, nprices / 2, &param1, &param2);
    }

    else {
      fclose(fp);
      printf("\n\nInvalid variable (%s)  Press any key...", lptr);
      goto FINISH;
    }

    /*
       Compute, analyze, and save the variable
    */

    vptr = vars[nvars] = (double *)MALLOC(nprices * sizeof(double));
    if (vars[nvars] == NULL) {
      fclose(fp);
      printf("\n\nInsufficient memory reading market script file %s  Press any "
             "key...",
             ScriptName);
      goto FINISH;
    } // If insufficient memory

    ret_val =
        comp_var(nprices, var, param1, param2, param3, param4, open1, high1,
                 low1, close1, volume1, open2, high2, low2, close2, volume2,
                 &n_done, &first_date, &last_date, vptr, work1, work2, work3);
    if (ret_val) {
      printf("\n\nERROR processing variable script file... Aborting");
      goto FINISH;
    }

    if (nprices - n_done >
        front_bad) // Keep track of max invalid at start of series
      front_bad = nprices - n_done;

    var_ent = entropy(last_date - first_date + 1, vptr + first_date);
    basic_stats(last_date - first_date + 1, vptr + first_date, var_work,
                &var_mean, &var_min, &var_max, &var_iqr);
    printf("\n%4d %15s %7d      %8d     %8d     %9.3lf     %9.3lf     %9.3lf  "
           "%9.3lf    %8.3lf     %.3lf",
           nvars + 1, user_name, n_done, date1[first_date], date1[last_date],
           var_mean, var_min, var_max, var_iqr,
           (var_max - var_min) / (var_iqr + 1.e-60), var_ent);

    ++nvars;
  } // For all script lines

  fclose(fp);

  printf("\nFinished processing variable script file.  Writing output file...");

  /*
  -------------------------------------------------------------------------------

     All variables are computed.  Write the output file.

  -------------------------------------------------------------------------------
  */

  if (fopen_s(&fp, "OUTVARS.TXT", "wt")) {
    printf("\n\nCannot open variable output file OUTVARS.TXT");
    goto FINISH;
  }

  // Write the header of variable names

  fprintf(fp, "Date");
  for (i = 0; i < nvars; i++) {
    fprintf(fp, " %s", var_names[i]);
    if (ferror(fp)) {
      printf("\nError writing variable output file OUTVARS.TXT");
      goto FINISH;
    }
  }

  // Write the cases

  for (icase = front_bad; icase < nprices; icase++) {
    fprintf(fp, "\n%8d", date1[icase]);
    for (i = 0; i < nvars; i++) {
      fprintf(fp, " %10.5lf", vars[i][icase]);
      if (ferror(fp)) {
        printf("\nError writing variable output file OUTVARS.TXT");
        goto FINISH;
      }
    } // For variables
  } // For cases

  fclose(fp);
  printf("\nSuccess");

FINISH:
  printf("\n\nPress any key...");
  _getch(); // Wait for user to press a key

  if (date1 != NULL)
    FREE(date1);
  if (open1 != NULL)
    FREE(open1);
  if (high1 != NULL)
    FREE(high1);
  if (low1 != NULL)
    FREE(low1);
  if (close1 != NULL)
    FREE(close1);
  if (volume1 != NULL)
    FREE(volume1);
  if (date2 != NULL)
    FREE(date2);
  if (open2 != NULL)
    FREE(open2);
  if (high2 != NULL)
    FREE(high2);
  if (low2 != NULL)
    FREE(low2);
  if (close2 != NULL)
    FREE(close2);
  if (volume2 != NULL)
    FREE(volume2);
  if (var_work != NULL)
    FREE(var_work);
  if (work1 != NULL)
    FREE(work1);
  if (work2 != NULL)
    FREE(work2);
  if (work3 != NULL)
    FREE(work3);
  for (i = 0; i < nvars; i++) {
    if (vars[i] != NULL)
      FREE(vars[i]);
  }

#if MEMDEBUG
  memclose();
#endif

  exit(0);
}
