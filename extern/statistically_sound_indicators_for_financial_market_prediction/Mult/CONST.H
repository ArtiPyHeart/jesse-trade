// Constants for the program

/*
   These interpose memory allocation checking in MEM64.CPP.
   To use ordinary malloc/free set MEMDEBUG to 0
*/

#define MEMDEBUG 1

#if MEMDEBUG
#define MALLOC memalloc
#define FREE memfree
#define REALLOC memrealloc
#define MEMTEXT memtext
#define MEMCLOSE memclose
#else
#define MALLOC malloc
#define FREE free
#define REALLOC realloc
#define MEMTEXT notext
#define MEMCLOSE nomemclose
#endif

/*
   Constants
*/

#if !defined(MAXPOSNUM)
#define MAXPOSNUM 2147483647
#endif
#if !defined(MAXNEGNUM)
#define MAXNEGNUM -2147483647 /* Actually is -2147483648 */
#endif
#if !defined(PI)
#define PI 3.141592653589793
#endif

#define KEY_ESCAPE 27
#define KEY_CTRLQ 17 /* Total abort */

/*
   These are universal error codes.  The first two have mandatory values!
*/

#define ERROR_OK 0
#define ERROR_NO_ERROR 0
#define ERROR_ESCAPE 1
#define ERROR_ABORT 2
#define ERROR_INSUFFICIENT_MEMORY 3
#define ERROR_SYNTAX 4
#define ERROR_FILE 5

/*
   Assorted constants
*/

/*
   Variables
*/

#define VAR_TREND_RANK 10
#define VAR_CMMA_RANK 12
#define VAR_TREND_MEDIAN 14
#define VAR_CMMA_MEDIAN 16
#define VAR_TREND_RANGE 18
#define VAR_CMMA_RANGE 20
#define VAR_TREND_IQR 22
#define VAR_CMMA_IQR 24
#define VAR_TREND_CLUMP 26
#define VAR_CMMA_CLUMP 28
#define VAR_MAHAL 50
#define VAR_ABS_RATIO 52
#define VAR_ABS_SHIFT 54
#define VAR_COHERENCE 56
#define VAR_DELTA_COHERENCE 58
#define VAR_JANUS_INDEX_MARKET 80
#define VAR_JANUS_INDEX_DOM 82
#define VAR_JANUS_RAW_RS 84
#define VAR_JANUS_FRACTILE_RS 86
#define VAR_JANUS_DELTA_FRACTILE_RS 88
#define VAR_JANUS_RSS 90
#define VAR_JANUS_DELTA_RSS 92
#define VAR_JANUS_DOM 94
#define VAR_JANUS_DOE 96
#define VAR_JANUS_RAW_RM 98
#define VAR_JANUS_FRACTILE_RM 100
#define VAR_JANUS_DELTA_FRACTILE_RM 102
#define VAR_JANUS_RS_LEADER_EQUITY 104
#define VAR_JANUS_RS_LAGGARD_EQUITY 106
#define VAR_JANUS_RS_PS 108
#define VAR_JANUS_RS_LEADER_ADVANTAGE 110
#define VAR_JANUS_RS_LAGGARD_ADVANTAGE 112
#define VAR_JANUS_RM_LEADER_EQUITY 114
#define VAR_JANUS_RM_LAGGARD_EQUITY 116
#define VAR_JANUS_RM_PS 118
#define VAR_JANUS_RM_LEADER_ADVANTAGE 120
#define VAR_JANUS_RM_LAGGARD_ADVANTAGE 122
#define VAR_JANUS_CMA_OOS 130
#define VAR_JANUS_LEADER_CMA_OOS 132
#define VAR_JANUS_OOS_AVG 150

/*
   Program limitations
*/

#define MAX_NAME_LENGTH 15
#define MAX_MARKETS 1024
#define MAX_VARS 8192