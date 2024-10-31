#include "class_d_acc.h"
#include <stddef.h>
#include <float.h>
#include "mwmathutil.h"
#include "rtwtypes.h"
#include "class_d_acc_types.h"
#include "class_d_acc_private.h"
#include <stdio.h>
#include "slexec_vm_simstruct_bridge.h"
#include "slexec_vm_zc_functions.h"
#include "slexec_vm_lookup_functions.h"
#include "slsv_diagnostic_codegen_c_api.h"
#include "simtarget/slSimTgtMdlrefSfcnBridge.h"
#include "simstruc.h"
#include "fixedpoint.h"
#define CodeFormat S-Function
#define AccDefine1 Accelerator_S-Function
#include "simtarget/slAccSfcnBridge.h"
#ifndef __RTW_UTFREE__  
extern void * utMalloc ( size_t ) ; extern void utFree ( void * ) ;
#endif
boolean_T class_d_acc_rt_TDelayUpdateTailOrGrowBuf ( int_T * bufSzPtr , int_T
* tailPtr , int_T * headPtr , int_T * lastPtr , real_T tMinusDelay , real_T *
* uBufPtr , boolean_T isfixedbuf , boolean_T istransportdelay , int_T *
maxNewBufSzPtr ) { int_T testIdx ; int_T tail = * tailPtr ; int_T bufSz = *
bufSzPtr ; real_T * tBuf = * uBufPtr + bufSz ; real_T * xBuf = ( NULL ) ;
int_T numBuffer = 2 ; if ( istransportdelay ) { numBuffer = 3 ; xBuf = *
uBufPtr + 2 * bufSz ; } testIdx = ( tail < ( bufSz - 1 ) ) ? ( tail + 1 ) : 0
; if ( ( tMinusDelay <= tBuf [ testIdx ] ) && ! isfixedbuf ) { int_T j ;
real_T * tempT ; real_T * tempU ; real_T * tempX = ( NULL ) ; real_T * uBuf =
* uBufPtr ; int_T newBufSz = bufSz + 1024 ; if ( newBufSz > * maxNewBufSzPtr
) { * maxNewBufSzPtr = newBufSz ; } tempU = ( real_T * ) utMalloc ( numBuffer
* newBufSz * sizeof ( real_T ) ) ; if ( tempU == ( NULL ) ) { return ( false
) ; } tempT = tempU + newBufSz ; if ( istransportdelay ) tempX = tempT +
newBufSz ; for ( j = tail ; j < bufSz ; j ++ ) { tempT [ j - tail ] = tBuf [
j ] ; tempU [ j - tail ] = uBuf [ j ] ; if ( istransportdelay ) tempX [ j -
tail ] = xBuf [ j ] ; } for ( j = 0 ; j < tail ; j ++ ) { tempT [ j + bufSz -
tail ] = tBuf [ j ] ; tempU [ j + bufSz - tail ] = uBuf [ j ] ; if ( istransportdelay ) tempX [ j + bufSz - tail ] = xBuf [ j ] ; } if ( * lastPtr > tail ) { * lastPtr -= tail ; } else { * lastPtr += ( bufSz - tail ) ; } * tailPtr = 0 ; * headPtr = bufSz ; utFree ( uBuf ) ; * bufSzPtr = newBufSz ; * uBufPtr = tempU ; } else { * tailPtr = testIdx ; } return ( true ) ; } real_T class_d_acc_rt_TDelayInterpolate ( real_T tMinusDelay , real_T tStart , real_T * uBuf , int_T bufSz , int_T * lastIdx , int_T oldestIdx , int_T newIdx , real_T initOutput , boolean_T discrete , boolean_T minorStepAndTAtLastMajorOutput ) { int_T i ; real_T yout , t1 , t2 , u1 , u2 ; real_T * tBuf = uBuf + bufSz ; if ( ( newIdx == 0 ) && ( oldestIdx == 0 ) && ( tMinusDelay > tStart ) ) return initOutput ; if ( tMinusDelay <= tStart ) return initOutput ; if ( ( tMinusDelay <= tBuf [ oldestIdx ] ) ) { if ( discrete ) { return ( uBuf [ oldestIdx ] ) ; } else { int_T tempIdx = oldestIdx + 1 ; if ( oldestIdx == bufSz - 1 ) tempIdx = 0 ; t1 = tBuf [ oldestIdx ] ; t2 = tBuf [ tempIdx ] ; u1 = uBuf [ oldestIdx ] ; u2 = uBuf [ tempIdx ] ; if ( t2 == t1 ) { if ( tMinusDelay >= t2 ) { yout = u2 ; } else { yout = u1 ; } } else { real_T f1 = ( t2 - tMinusDelay ) / ( t2 - t1 ) ; real_T f2 = 1.0 - f1 ; yout = f1 * u1 + f2 * u2 ; } return yout ; } } if ( minorStepAndTAtLastMajorOutput ) { if ( newIdx != 0 ) { if ( * lastIdx == newIdx ) { ( * lastIdx ) -- ; } newIdx -- ; } else { if ( * lastIdx == newIdx ) { * lastIdx = bufSz - 1 ; } newIdx = bufSz - 1 ; } } i = * lastIdx ; if ( tBuf [ i ] < tMinusDelay ) { while ( tBuf [ i ] < tMinusDelay ) { if ( i == newIdx ) break ; i = ( i < ( bufSz - 1 ) ) ? ( i + 1 ) : 0 ; } } else { while ( tBuf [ i ] >= tMinusDelay ) { i = ( i > 0 ) ? i - 1 : ( bufSz - 1 ) ; } i = ( i < ( bufSz - 1 ) ) ? ( i + 1 ) : 0 ; } * lastIdx = i ; if ( discrete ) { double tempEps = ( DBL_EPSILON ) * 128.0 ; double localEps = tempEps * muDoubleScalarAbs ( tBuf [ i ] ) ; if ( tempEps > localEps ) { localEps = tempEps ; } localEps = localEps / 2.0 ; if ( tMinusDelay >= ( tBuf [ i ] - localEps ) ) { yout = uBuf [ i ] ; } else { if ( i == 0 ) { yout = uBuf [ bufSz - 1 ] ; } else { yout = uBuf [ i - 1 ] ; } } } else { if ( i == 0 ) { t1 = tBuf [ bufSz - 1 ] ; u1 = uBuf [ bufSz - 1 ] ; } else { t1 = tBuf [ i - 1 ] ; u1 = uBuf [ i - 1 ] ; } t2 = tBuf [ i ] ; u2 = uBuf [ i ] ; if ( t2 == t1 ) { if ( tMinusDelay >= t2 ) { yout = u2 ; } else { yout = u1 ; } } else { real_T f1 = ( t2 - tMinusDelay ) / ( t2 - t1 ) ; real_T f2 = 1.0 - f1 ; yout = f1 * u1 + f2 * u2 ; } } return ( yout ) ; } void rt_ssGetBlockPath ( void * S , int_T sysIdx , int_T blkIdx , char_T * * path ) { _ssGetBlockPath ( ( SimStruct * ) S , sysIdx , blkIdx , path ) ; } void rt_ssSet_slErrMsg ( void * S , void * diag ) { SimStruct * castedS = ( SimStruct * ) S ; if ( ! _ssIsErrorStatusAslErrMsg ( castedS ) ) { _ssSet_slErrMsg ( castedS , diag ) ; } else { _ssDiscardDiagnostic ( castedS , diag ) ; } } void rt_ssReportDiagnosticAsWarning ( void * S , void * diag ) { _ssReportDiagnosticAsWarning ( ( SimStruct * ) S , diag ) ; } void rt_ssReportDiagnosticAsInfo ( void * S , void * diag ) { _ssReportDiagnosticAsInfo ( ( SimStruct * ) S , diag ) ; } static void mdlOutputs ( SimStruct * S , int_T tid ) { B_class_d_T * _rtB ; DW_class_d_T * _rtDW ; P_class_d_T * _rtP ; X_class_d_T * _rtX ; int32_T isHit ; _rtDW = ( ( DW_class_d_T * ) ssGetRootDWork ( S ) ) ; _rtX = ( ( X_class_d_T * ) ssGetContStates ( S ) ) ; _rtP = ( ( P_class_d_T * ) ssGetModelRtp ( S ) ) ; _rtB = ( ( B_class_d_T * ) _ssGetModelBlockIO ( S ) ) ; _rtB -> B_0_0_0 = 0.0 ; _rtB -> B_0_0_0 += _rtP -> P_1 [ 0 ] * _rtX -> OutputFilter_CSTATE [ 0 ] ; _rtB -> B_0_0_0 += _rtP -> P_1 [ 1 ] * _rtX -> OutputFilter_CSTATE [ 1 ] ; _rtB -> B_0_1_8 = 0.0 ; _rtB -> B_0_1_8 += _rtP -> P_4 [ 0 ] * _rtX -> ZeroPole_CSTATE [ 0 ] ; _rtB -> B_0_1_8 += _rtP -> P_4 [ 1 ] * _rtX -> ZeroPole_CSTATE [ 1 ] ; _rtB -> B_0_1_8 += _rtP -> P_5 * _rtB -> B_0_0_0 ; { real_T * * uBuffer = ( real_T * * ) & _rtDW -> TransportDelay_PWORK . TUbufferPtrs [ 0 ] ; real_T simTime = ssGetT ( S ) ; real_T tMinusDelay = simTime - _rtP -> P_6 ; _rtB -> B_0_2_16 = class_d_acc_rt_TDelayInterpolate ( tMinusDelay , 0.0 , * uBuffer , _rtDW -> TransportDelay_IWORK . CircularBufSize , & _rtDW -> TransportDelay_IWORK . Last , _rtDW -> TransportDelay_IWORK . Tail , _rtDW -> TransportDelay_IWORK . Head , _rtP -> P_7 , 0 , ( boolean_T ) ( ssIsMinorTimeStep ( S ) && ( ( * uBuffer + _rtDW -> TransportDelay_IWORK . CircularBufSize ) [ _rtDW -> TransportDelay_IWORK . Head ] == ssGetT ( S ) ) ) ) ; } isHit = ssIsSampleHit ( S , 1 , 0 ) ; if ( isHit != 0 ) { if ( ssIsModeUpdateTimeStep ( S ) ) { _rtDW -> Compare_Mode = ( _rtB -> B_0_2_16 <= _rtB -> B_0_9_72 ) ; } _rtB -> B_0_3_24 = ( ( real_T ) _rtDW -> Compare_Mode + _rtP -> P_8 ) * _rtP -> P_9 ; } ssCallAccelRunBlock ( S , 0 , 7 , SS_CALL_MDL_OUTPUTS ) ; { if ( _rtDW -> _asyncqueue_inserted_for_ContinuousOutput_PWORK . AQHandles && ssGetLogOutput ( S ) ) { sdiWriteSignal ( _rtDW -> _asyncqueue_inserted_for_ContinuousOutput_PWORK . AQHandles , ssGetTaskTime ( S , 0 ) , ( char * ) & _rtB -> B_0_0_0 + 0 ) ; } } isHit = ssIsSampleHit ( S , 2 , 0 ) ; if ( isHit != 0 ) { { if ( _rtDW -> _asyncqueue_inserted_for_SampledOutput_PWORK . AQHandles && ssGetLogOutput ( S ) ) { sdiWriteSignal ( _rtDW -> _asyncqueue_inserted_for_SampledOutput_PWORK . AQHandles , ssGetTaskTime ( S , 2 ) , ( char * ) & _rtB -> B_0_0_0 + 0 ) ; } } } _rtB -> B_0_4_32 = _rtP -> P_10 * _rtB -> B_0_0_0 ; _rtB -> B_0_5_40 = _rtB -> B_0_4_32 - ( muDoubleScalarSin ( _rtP -> P_13 * ssGetTaskTime ( S , 0 ) + _rtP -> P_14 ) * _rtP -> P_11 + _rtP -> P_12 ) ; _rtB -> B_0_6_48 = _rtP -> P_15 * _rtB -> B_0_5_40 ; _rtB -> B_0_7_56 = 0.0 ; _rtB -> B_0_7_56 += _rtP -> P_17 [ 0 ] * _rtX -> Integrator_CSTATE [ 0 ] ; _rtB -> B_0_7_56 += _rtP -> P_17 [ 1 ] * _rtX -> Integrator_CSTATE [ 1 ] ; _rtB -> B_0_7_56 += _rtP -> P_17 [ 2 ] * _rtX -> Integrator_CSTATE [ 2 ] ; _rtB -> B_0_8_64 = _rtB -> B_0_1_8 + _rtB -> B_0_7_56 ; UNUSED_PARAMETER ( tid ) ; } static void mdlOutputsTID3 ( SimStruct * S , int_T tid ) { B_class_d_T * _rtB ; P_class_d_T * _rtP ; _rtP = ( ( P_class_d_T * ) ssGetModelRtp ( S ) ) ; _rtB = ( ( B_class_d_T * ) _ssGetModelBlockIO ( S ) ) ; _rtB -> B_0_9_72 = _rtP -> P_18 ; UNUSED_PARAMETER ( tid ) ; }
#define MDL_UPDATE
static void mdlUpdate ( SimStruct * S , int_T tid ) { B_class_d_T * _rtB ;
DW_class_d_T * _rtDW ; P_class_d_T * _rtP ; _rtDW = ( ( DW_class_d_T * )
ssGetRootDWork ( S ) ) ; _rtP = ( ( P_class_d_T * ) ssGetModelRtp ( S ) ) ;
_rtB = ( ( B_class_d_T * ) _ssGetModelBlockIO ( S ) ) ; { real_T * * uBuffer
= ( real_T * * ) & _rtDW -> TransportDelay_PWORK . TUbufferPtrs [ 0 ] ;
real_T simTime = ssGetT ( S ) ; _rtDW -> TransportDelay_IWORK . Head = ( ( _rtDW
-> TransportDelay_IWORK . Head < ( _rtDW -> TransportDelay_IWORK .
CircularBufSize - 1 ) ) ? ( _rtDW -> TransportDelay_IWORK . Head + 1 ) : 0 )
; if ( _rtDW -> TransportDelay_IWORK . Head == _rtDW -> TransportDelay_IWORK
. Tail ) { if ( ! class_d_acc_rt_TDelayUpdateTailOrGrowBuf ( & _rtDW ->
TransportDelay_IWORK . CircularBufSize , & _rtDW -> TransportDelay_IWORK .
Tail , & _rtDW -> TransportDelay_IWORK . Head , & _rtDW ->
TransportDelay_IWORK . Last , simTime - _rtP -> P_6 , uBuffer , ( boolean_T )
0 , false , & _rtDW -> TransportDelay_IWORK . MaxNewBufSize ) ) {
ssSetErrorStatus ( S , ( char_T * ) "\"tdelay memory allocation error\"" ) ;
return ; } } ( * uBuffer + _rtDW -> TransportDelay_IWORK . CircularBufSize )
[ _rtDW -> TransportDelay_IWORK . Head ] = simTime ; ( * uBuffer ) [ _rtDW ->
TransportDelay_IWORK . Head ] = _rtB -> B_0_8_64 ; } UNUSED_PARAMETER ( tid )
; }
#define MDL_UPDATE
static void mdlUpdateTID3 ( SimStruct * S , int_T tid ) { UNUSED_PARAMETER ( tid
) ; }
#define MDL_DERIVATIVES
static void mdlDerivatives ( SimStruct * S ) { B_class_d_T * _rtB ;
P_class_d_T * _rtP ; XDot_class_d_T * _rtXdot ; X_class_d_T * _rtX ; _rtXdot
= ( ( XDot_class_d_T * ) ssGetdX ( S ) ) ; _rtX = ( ( X_class_d_T * )
ssGetContStates ( S ) ) ; _rtP = ( ( P_class_d_T * ) ssGetModelRtp ( S ) ) ;
_rtB = ( ( B_class_d_T * ) _ssGetModelBlockIO ( S ) ) ; _rtXdot ->
OutputFilter_CSTATE [ 0 ] = 0.0 ; _rtXdot -> OutputFilter_CSTATE [ 0 ] +=
_rtP -> P_0 [ 0 ] * _rtX -> OutputFilter_CSTATE [ 0 ] ; _rtXdot ->
OutputFilter_CSTATE [ 1 ] = 0.0 ; _rtXdot -> OutputFilter_CSTATE [ 0 ] +=
_rtP -> P_0 [ 1 ] * _rtX -> OutputFilter_CSTATE [ 1 ] ; _rtXdot ->
OutputFilter_CSTATE [ 1 ] += _rtX -> OutputFilter_CSTATE [ 0 ] ; _rtXdot ->
OutputFilter_CSTATE [ 0 ] += _rtB -> B_0_3_24 ; _rtXdot -> ZeroPole_CSTATE [
0 ] = 0.0 ; _rtXdot -> ZeroPole_CSTATE [ 1 ] = 0.0 ; _rtXdot ->
ZeroPole_CSTATE [ 0 ] += _rtP -> P_2 [ 0 ] * _rtX -> ZeroPole_CSTATE [ 0 ] ;
_rtXdot -> ZeroPole_CSTATE [ 0 ] += _rtP -> P_2 [ 1 ] * _rtX ->
ZeroPole_CSTATE [ 1 ] ; _rtXdot -> ZeroPole_CSTATE [ 1 ] += _rtX ->
ZeroPole_CSTATE [ 0 ] * _rtP -> P_2 [ 2 ] ; _rtXdot -> ZeroPole_CSTATE [ 0 ]
+= _rtP -> P_3 * _rtB -> B_0_0_0 ; _rtXdot -> Integrator_CSTATE [ 0 ] = 0.0 ;
_rtXdot -> Integrator_CSTATE [ 0 ] += _rtP -> P_16 [ 0 ] * _rtX ->
Integrator_CSTATE [ 0 ] ; _rtXdot -> Integrator_CSTATE [ 1 ] = 0.0 ; _rtXdot
-> Integrator_CSTATE [ 0 ] += _rtP -> P_16 [ 1 ] * _rtX -> Integrator_CSTATE
[ 1 ] ; _rtXdot -> Integrator_CSTATE [ 2 ] = 0.0 ; _rtXdot ->
Integrator_CSTATE [ 0 ] += _rtP -> P_16 [ 2 ] * _rtX -> Integrator_CSTATE [ 2
] ; _rtXdot -> Integrator_CSTATE [ 1 ] += _rtX -> Integrator_CSTATE [ 0 ] ;
_rtXdot -> Integrator_CSTATE [ 2 ] += _rtX -> Integrator_CSTATE [ 1 ] ;
_rtXdot -> Integrator_CSTATE [ 0 ] += _rtB -> B_0_6_48 ; }
#define MDL_ZERO_CROSSINGS
static void mdlZeroCrossings ( SimStruct * S ) { B_class_d_T * _rtB ;
ZCV_class_d_T * _rtZCSV ; _rtZCSV = ( ( ZCV_class_d_T * )
ssGetSolverZcSignalVector ( S ) ) ; _rtB = ( ( B_class_d_T * )
_ssGetModelBlockIO ( S ) ) ; _rtZCSV -> Compare_RelopInput_ZC = _rtB ->
B_0_2_16 - _rtB -> B_0_9_72 ; } static void mdlInitializeSizes ( SimStruct *
S ) { ssSetChecksumVal ( S , 0 , 2817727775U ) ; ssSetChecksumVal ( S , 1 ,
869350267U ) ; ssSetChecksumVal ( S , 2 , 1370916429U ) ; ssSetChecksumVal ( S
, 3 , 608773686U ) ; { mxArray * slVerStructMat = ( NULL ) ; mxArray *
slStrMat = mxCreateString ( "simulink" ) ; char slVerChar [ 10 ] ; int status
= mexCallMATLAB ( 1 , & slVerStructMat , 1 , & slStrMat , "ver" ) ; if ( status
== 0 ) { mxArray * slVerMat = mxGetField ( slVerStructMat , 0 , "Version" ) ;
if ( slVerMat == ( NULL ) ) { status = 1 ; } else { status = mxGetString ( slVerMat , slVerChar , 10 ) ; } } mxDestroyArray ( slStrMat ) ; mxDestroyArray ( slVerStructMat ) ; if ( ( status == 1 ) || ( strcmp ( slVerChar , "24.2" ) != 0 ) ) { return ; } } ssSetOptions ( S , SS_OPTION_EXCEPTION_FREE_CODE ) ; if ( ssGetSizeofDWork ( S ) != ( SLSize ) sizeof ( DW_class_d_T ) ) { static char msg [ 256 ] ; snprintf ( msg , 256 , "Unexpected error: Internal DWork sizes do " "not match for accelerator mex file (%ld vs %lu)." , ( signed long ) ssGetSizeofDWork ( S ) , ( unsigned long ) sizeof ( DW_class_d_T ) ) ; ssSetErrorStatus ( S , msg ) ; } if ( ssGetSizeofGlobalBlockIO ( S ) != ( SLSize ) sizeof ( B_class_d_T ) ) { static char msg [ 256 ] ; snprintf ( msg , 256 , "Unexpected error: Internal BlockIO sizes do " "not match for accelerator mex file (%ld vs %lu)." , ( signed long ) ssGetSizeofGlobalBlockIO ( S ) , ( unsigned long ) sizeof ( B_class_d_T ) ) ; ssSetErrorStatus ( S , msg ) ; } { int ssSizeofParams ; ssGetSizeofParams ( S , & ssSizeofParams ) ; if ( ssSizeofParams != sizeof ( P_class_d_T ) ) { static char msg [ 256 ] ; snprintf ( msg , 256 , "Unexpected error: Internal Parameters sizes do " "not match for accelerator mex file (%d vs %lu)." , ssSizeofParams , ( unsigned long ) sizeof ( P_class_d_T ) ) ; ssSetErrorStatus ( S , msg ) ; } } _ssSetModelRtp ( S , ( real_T * ) & class_d_rtDefaultP ) ; } static void mdlInitializeSampleTimes ( SimStruct * S ) { slAccRegPrmChangeFcn ( S , mdlOutputsTID3 ) ; } static void mdlTerminate ( SimStruct * S ) { }
#include "simulink.c"
