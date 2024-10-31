#ifndef class_d_h_
#define class_d_h_
#ifndef class_d_COMMON_INCLUDES_
#define class_d_COMMON_INCLUDES_
#include <stdio.h>
#include <stdlib.h>
#include "sl_AsyncioQueue/AsyncioQueueCAPI.h"
#include "rtwtypes.h"
#include "sigstream_rtw.h"
#include "simtarget/slSimTgtSigstreamRTW.h"
#include "simtarget/slSimTgtSlioCoreRTW.h"
#include "simtarget/slSimTgtSlioClientsRTW.h"
#include "simtarget/slSimTgtSlioSdiRTW.h"
#include "simstruc.h"
#include "fixedpoint.h"
#include "raccel.h"
#include "slsv_diagnostic_codegen_c_api.h"
#include "rt_logging_simtarget.h"
#include "rt_nonfinite.h"
#include "math.h"
#include "dt_info.h"
#include "ext_work.h"
#endif
#include "class_d_types.h"
#include <stddef.h>
#include "rtw_modelmap_simtarget.h"
#include "rt_defines.h"
#include <string.h>
#include "rtGetInf.h"
#define MODEL_NAME class_d
#define NSAMPLE_TIMES (3) 
#define NINPUTS (0)       
#define NOUTPUTS (0)     
#define NBLOCKIO (9) 
#define NUM_ZC_EVENTS (0) 
#ifndef NCSTATES
#define NCSTATES (10)   
#elif NCSTATES != 10
#error Invalid specification of NCSTATES defined in compiler command
#endif
#ifndef rtmGetDataMapInfo
#define rtmGetDataMapInfo(rtm) (*rt_dataMapInfoPtr)
#endif
#ifndef rtmSetDataMapInfo
#define rtmSetDataMapInfo(rtm, val) (rt_dataMapInfoPtr = &val)
#endif
#ifndef IN_RACCEL_MAIN
#endif
typedef struct { real_T opyihwpyem ; real_T kdfvhapu1i ; real_T aemqx1t5mh ;
real_T kzsfgqhpv1 ; real_T lwhumkefgn ; real_T odv4zor5wc ; real_T g1gyqhmuvp
; real_T ozgjv5kcoq ; real_T iotcjfxadz ; } B ; typedef struct { struct {
real_T modelTStart ; } nequhacir4 ; struct { void * TUbufferPtrs [ 2 ] ; }
etzcdtgg4v ; struct { void * LoggedData [ 3 ] ; } h4urlizny4 ; struct { void
* AQHandles ; } k3ahccqiwo ; struct { void * AQHandles ; } ozcp50hell ;
struct { int_T Tail ; int_T Head ; int_T Last ; int_T CircularBufSize ; int_T
MaxNewBufSize ; } paeebttgho ; boolean_T e4drogkh1r ; } DW ; typedef struct {
real_T n2zca1rgni [ 2 ] ; real_T az3rcuqa4j [ 4 ] ; real_T kgoujacfja [ 4 ] ;
} X ; typedef struct { real_T n2zca1rgni [ 2 ] ; real_T az3rcuqa4j [ 4 ] ;
real_T kgoujacfja [ 4 ] ; } XDot ; typedef struct { boolean_T n2zca1rgni [ 2
] ; boolean_T az3rcuqa4j [ 4 ] ; boolean_T kgoujacfja [ 4 ] ; } XDis ;
typedef struct { real_T n2zca1rgni [ 2 ] ; real_T az3rcuqa4j [ 4 ] ; real_T
kgoujacfja [ 4 ] ; } CStateAbsTol ; typedef struct { real_T n2zca1rgni [ 2 ]
; real_T az3rcuqa4j [ 4 ] ; real_T kgoujacfja [ 4 ] ; } CXPtMin ; typedef
struct { real_T n2zca1rgni [ 2 ] ; real_T az3rcuqa4j [ 4 ] ; real_T
kgoujacfja [ 4 ] ; } CXPtMax ; typedef struct { real_T phbximogki ; } ZCV ;
typedef struct { rtwCAPI_ModelMappingInfo mmi ; } DataMapInfo ; struct P_ {
real_T CompareToConstant_const ; real_T OutputFilter_A [ 2 ] ; real_T
OutputFilter_C [ 2 ] ; real_T ZeroPole_A [ 8 ] ; real_T ZeroPole_B [ 2 ] ;
real_T ZeroPole_C [ 4 ] ; real_T ZeroPole_D ; real_T TransportDelay_Delay ;
real_T TransportDelay_InitOutput ; real_T Bias_Bias ; real_T Gain_Gain ;
real_T Gain_Gain_idyf3glywd ; real_T SineWave_Amp ; real_T SineWave_Bias ;
real_T SineWave_Freq ; real_T SineWave_Phase ; real_T Gain1_Gain ; real_T
Integrator_A [ 4 ] ; real_T Integrator_C [ 4 ] ; } ; extern const char_T *
RT_MEMORY_ALLOCATION_ERROR ; extern B rtB ; extern X rtX ; extern DW rtDW ;
extern P rtP ; extern mxArray * mr_class_d_GetDWork ( ) ; extern void
mr_class_d_SetDWork ( const mxArray * ssDW ) ; extern mxArray *
mr_class_d_GetSimStateDisallowedBlocks ( ) ; extern const
rtwCAPI_ModelMappingStaticInfo * class_d_GetCAPIStaticMap ( void ) ; extern
SimStruct * const rtS ; extern DataMapInfo * rt_dataMapInfoPtr ; extern
rtwCAPI_ModelMappingInfo * rt_modelMapInfoPtr ; void MdlOutputs ( int_T tid )
; void MdlOutputsParameterSampleTime ( int_T tid ) ; void MdlUpdate ( int_T
tid ) ; void MdlTerminate ( void ) ; void MdlInitializeSizes ( void ) ; void
MdlInitializeSampleTimes ( void ) ; SimStruct * raccel_register_model ( ssExecutionInfo * executionInfo ) ;
#endif
