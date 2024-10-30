#ifndef class_d_acc_h_
#define class_d_acc_h_
#ifndef class_d_acc_COMMON_INCLUDES_
#define class_d_acc_COMMON_INCLUDES_
#include <stdlib.h>
#define S_FUNCTION_NAME simulink_only_sfcn
#define S_FUNCTION_LEVEL 2
#ifndef RTW_GENERATED_S_FUNCTION
#define RTW_GENERATED_S_FUNCTION
#endif
#include "sl_AsyncioQueue/AsyncioQueueCAPI.h"
#include "rtwtypes.h"
#include "simstruc.h"
#include "fixedpoint.h"
#include "math.h"
#endif
#include "class_d_acc_types.h"
#include "rt_defines.h"
#include <stddef.h>
typedef struct { real_T B_0_0_0 ; real_T B_0_1_8 ; real_T B_0_2_16 ; real_T
B_0_3_24 ; real_T B_0_4_32 ; real_T B_0_5_40 ; real_T B_0_6_48 ; real_T
B_0_7_56 ; real_T B_0_8_64 ; } B_class_d_T ; typedef struct { struct { real_T
modelTStart ; } TransportDelay_RWORK ; struct { void * TUbufferPtrs [ 2 ] ; }
TransportDelay_PWORK ; void * Scope_PWORK [ 3 ] ; struct { void * AQHandles ;
} _asyncqueue_inserted_for_ContinuousOutput_PWORK ; struct { void * AQHandles
; } _asyncqueue_inserted_for_SampledOutput_PWORK ; struct { int_T Tail ;
int_T Head ; int_T Last ; int_T CircularBufSize ; int_T MaxNewBufSize ; }
TransportDelay_IWORK ; boolean_T Compare_Mode ; char_T pad_Compare_Mode [ 3 ]
; } DW_class_d_T ; typedef struct { real_T OutputFilter_CSTATE [ 2 ] ; real_T
ZeroPole_CSTATE [ 2 ] ; real_T Integrator_CSTATE [ 2 ] ; } X_class_d_T ;
typedef struct { real_T OutputFilter_CSTATE [ 2 ] ; real_T ZeroPole_CSTATE [
2 ] ; real_T Integrator_CSTATE [ 2 ] ; } XDot_class_d_T ; typedef struct {
boolean_T OutputFilter_CSTATE [ 2 ] ; boolean_T ZeroPole_CSTATE [ 2 ] ;
boolean_T Integrator_CSTATE [ 2 ] ; } XDis_class_d_T ; typedef struct {
real_T OutputFilter_CSTATE [ 2 ] ; real_T ZeroPole_CSTATE [ 2 ] ; real_T
Integrator_CSTATE [ 2 ] ; } CStateAbsTol_class_d_T ; typedef struct { real_T
OutputFilter_CSTATE [ 2 ] ; real_T ZeroPole_CSTATE [ 2 ] ; real_T
Integrator_CSTATE [ 2 ] ; } CXPtMin_class_d_T ; typedef struct { real_T
OutputFilter_CSTATE [ 2 ] ; real_T ZeroPole_CSTATE [ 2 ] ; real_T
Integrator_CSTATE [ 2 ] ; } CXPtMax_class_d_T ; typedef struct { real_T
Compare_RelopInput_ZC ; } ZCV_class_d_T ; typedef struct { ZCSigState
Compare_RelopInput_ZCE ; } PrevZCX_class_d_T ; struct P_class_d_T_ { real_T
P_0 [ 2 ] ; real_T P_1 [ 2 ] ; real_T P_2 [ 3 ] ; real_T P_3 ; real_T P_4 [ 2
] ; real_T P_5 ; real_T P_6 ; real_T P_7 ; real_T P_8 ; real_T P_9 ; real_T
P_10 ; real_T P_11 [ 2 ] ; real_T P_12 [ 2 ] ; real_T P_13 ; real_T P_14 ;
real_T P_15 ; real_T P_16 ; real_T P_17 ; } ; extern P_class_d_T
class_d_rtDefaultP ;
#endif
