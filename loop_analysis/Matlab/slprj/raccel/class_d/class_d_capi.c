#include "rtw_capi.h"
#ifdef HOST_CAPI_BUILD
#include "class_d_capi_host.h"
#define sizeof(...) ((size_t)(0xFFFF))
#undef rt_offsetof
#define rt_offsetof(s,el) ((uint16_T)(0xFFFF))
#define TARGET_CONST
#define TARGET_STRING(s) (s)
#ifndef SS_UINT64
#define SS_UINT64 17
#endif
#ifndef SS_INT64
#define SS_INT64 18
#endif
#else
#include "builtin_typeid_types.h"
#include "class_d.h"
#include "class_d_capi.h"
#include "class_d_private.h"
#ifdef LIGHT_WEIGHT_CAPI
#define TARGET_CONST
#define TARGET_STRING(s)               ((NULL))
#else
#define TARGET_CONST                   const
#define TARGET_STRING(s)               (s)
#endif
#endif
static const rtwCAPI_Signals rtBlockSignals [ ] = { { 0 , 0 , TARGET_STRING ( "class_d/Gain" ) , TARGET_STRING ( "" ) , 0 , 0 , 0 , 0 , 0 } , { 1 , 0 , TARGET_STRING ( "class_d/Gain1" ) , TARGET_STRING ( "" ) , 0 , 0 , 0 , 0 , 0 } , { 2 , 0 , TARGET_STRING ( "class_d/Sum" ) , TARGET_STRING ( "" ) , 0 , 0 , 0 , 0 , 0 } , { 3 , 0 , TARGET_STRING ( "class_d/Sum1" ) , TARGET_STRING ( "" ) , 0 , 0 , 0 , 0 , 0 } , { 4 , 0 , TARGET_STRING ( "class_d/Integrator" ) , TARGET_STRING ( "" ) , 0 , 0 , 0 , 0 , 0 } , { 5 , 0 , TARGET_STRING ( "class_d/Output Filter" ) , TARGET_STRING ( "Output" ) , 0 , 0 , 0 , 0 , 0 } , { 6 , 0 , TARGET_STRING ( "class_d/Transport Delay" ) , TARGET_STRING ( "" ) , 0 , 0 , 0 , 0 , 0 } , { 7 , 0 , TARGET_STRING ( "class_d/Zero-Pole" ) , TARGET_STRING ( "" ) , 0 , 0 , 0 , 0 , 0 } , { 8 , 0 , TARGET_STRING ( "class_d/Modulator/Gain" ) , TARGET_STRING ( "" ) , 0 , 0 , 0 , 0 , 1 } , { 0 , 0 , ( NULL ) , ( NULL ) , 0 , 0 , 0 , 0 , 0 } } ; static const rtwCAPI_BlockParameters rtBlockParameters [ ] = { { 9 , TARGET_STRING ( "class_d/Gain" ) , TARGET_STRING ( "Gain" ) , 0 , 0 , 0 } , { 10 , TARGET_STRING ( "class_d/Gain1" ) , TARGET_STRING ( "Gain" ) , 0 , 0 , 0 } , { 11 , TARGET_STRING ( "class_d/Sine Wave" ) , TARGET_STRING ( "Amplitude" ) , 0 , 0 , 0 } , { 12 , TARGET_STRING ( "class_d/Sine Wave" ) , TARGET_STRING ( "Bias" ) , 0 , 0 , 0 } , { 13 , TARGET_STRING ( "class_d/Sine Wave" ) , TARGET_STRING ( "Frequency" ) , 0 , 0 , 0 } , { 14 , TARGET_STRING ( "class_d/Sine Wave" ) , TARGET_STRING ( "Phase" ) , 0 , 0 , 0 } , { 15 , TARGET_STRING ( "class_d/Integrator" ) , TARGET_STRING ( "A" ) , 0 , 1 , 0 } , { 16 , TARGET_STRING ( "class_d/Integrator" ) , TARGET_STRING ( "C" ) , 0 , 2 , 0 } , { 17 , TARGET_STRING ( "class_d/Output Filter" ) , TARGET_STRING ( "A" ) , 0 , 3 , 0 } , { 18 , TARGET_STRING ( "class_d/Output Filter" ) , TARGET_STRING ( "C" ) , 0 , 4 , 0 } , { 19 , TARGET_STRING ( "class_d/Transport Delay" ) , TARGET_STRING ( "DelayTime" ) , 0 , 0 , 0 } , { 20 , TARGET_STRING ( "class_d/Transport Delay" ) , TARGET_STRING ( "InitialOutput" ) , 0 , 0 , 0 } , { 21 , TARGET_STRING ( "class_d/Zero-Pole" ) , TARGET_STRING ( "A" ) , 0 , 5 , 0 } , { 22 , TARGET_STRING ( "class_d/Zero-Pole" ) , TARGET_STRING ( "B" ) , 0 , 3 , 0 } , { 23 , TARGET_STRING ( "class_d/Zero-Pole" ) , TARGET_STRING ( "C" ) , 0 , 1 , 0 } , { 24 , TARGET_STRING ( "class_d/Zero-Pole" ) , TARGET_STRING ( "D" ) , 0 , 0 , 0 } , { 25 , TARGET_STRING ( "class_d/Modulator/Compare To Constant" ) , TARGET_STRING ( "const" ) , 0 , 0 , 0 } , { 26 , TARGET_STRING ( "class_d/Modulator/Bias" ) , TARGET_STRING ( "Bias" ) , 0 , 0 , 0 } , { 27 , TARGET_STRING ( "class_d/Modulator/Gain" ) , TARGET_STRING ( "Gain" ) , 0 , 0 , 0 } , { 0 , ( NULL ) , ( NULL ) , 0 , 0 , 0 } } ; static int_T rt_LoggedStateIdxList [ ] = { - 1 } ; static const rtwCAPI_Signals rtRootInputs [ ] = { { 0 , 0 , ( NULL ) , ( NULL ) , 0 , 0 , 0 , 0 , 0 } } ; static const rtwCAPI_Signals rtRootOutputs [ ] = { { 0 , 0 , ( NULL ) , ( NULL ) , 0 , 0 , 0 , 0 , 0 } } ; static const rtwCAPI_ModelParameters rtModelParameters [ ] = { { 0 , ( NULL ) , 0 , 0 , 0 } } ;
#ifndef HOST_CAPI_BUILD
static void * rtDataAddrMap [ ] = { & rtB . lwhumkefgn , & rtB . g1gyqhmuvp ,
& rtB . iotcjfxadz , & rtB . odv4zor5wc , & rtB . ozgjv5kcoq , & rtB .
opyihwpyem , & rtB . aemqx1t5mh , & rtB . kdfvhapu1i , & rtB . kzsfgqhpv1 , &
rtP . Gain_Gain_idyf3glywd , & rtP . Gain1_Gain , & rtP . SineWave_Amp , &
rtP . SineWave_Bias , & rtP . SineWave_Freq , & rtP . SineWave_Phase , & rtP
. Integrator_A [ 0 ] , & rtP . Integrator_C [ 0 ] , & rtP . OutputFilter_A [
0 ] , & rtP . OutputFilter_C [ 0 ] , & rtP . TransportDelay_Delay , & rtP .
TransportDelay_InitOutput , & rtP . ZeroPole_A [ 0 ] , & rtP . ZeroPole_B [ 0
] , & rtP . ZeroPole_C [ 0 ] , & rtP . ZeroPole_D , & rtP .
CompareToConstant_const , & rtP . Bias_Bias , & rtP . Gain_Gain , } ; static
int32_T * rtVarDimsAddrMap [ ] = { ( NULL ) } ;
#endif
static TARGET_CONST rtwCAPI_DataTypeMap rtDataTypeMap [ ] = { { "double" ,
"real_T" , 0 , 0 , sizeof ( real_T ) , ( uint8_T ) SS_DOUBLE , 0 , 0 , 0 } }
;
#ifdef HOST_CAPI_BUILD
#undef sizeof
#endif
static TARGET_CONST rtwCAPI_ElementMap rtElementMap [ ] = { { ( NULL ) , 0 ,
0 , 0 , 0 } , } ; static const rtwCAPI_DimensionMap rtDimensionMap [ ] = { {
rtwCAPI_SCALAR , 0 , 2 , 0 } , { rtwCAPI_VECTOR , 2 , 2 , 0 } , {
rtwCAPI_VECTOR , 4 , 2 , 0 } , { rtwCAPI_VECTOR , 6 , 2 , 0 } , {
rtwCAPI_VECTOR , 8 , 2 , 0 } , { rtwCAPI_VECTOR , 10 , 2 , 0 } } ; static
const uint_T rtDimensionArray [ ] = { 1 , 1 , 4 , 1 , 1 , 4 , 2 , 1 , 1 , 2 ,
8 , 1 } ; static const real_T rtcapiStoredFloats [ ] = { 0.0 , 1.0 } ; static
const rtwCAPI_FixPtMap rtFixPtMap [ ] = { { ( NULL ) , ( NULL ) ,
rtwCAPI_FIX_RESERVED , 0 , 0 , ( boolean_T ) 0 } , } ; static const
rtwCAPI_SampleTimeMap rtSampleTimeMap [ ] = { { ( const void * ) &
rtcapiStoredFloats [ 0 ] , ( const void * ) & rtcapiStoredFloats [ 0 ] , ( int8_T ) 0 , ( uint8_T ) 0 } , { ( const void * ) & rtcapiStoredFloats [ 0 ] , ( const void * ) & rtcapiStoredFloats [ 1 ] , ( int8_T ) 1 , ( uint8_T ) 0 } } ; static rtwCAPI_ModelMappingStaticInfo mmiStatic = { { rtBlockSignals , 9 , rtRootInputs , 0 , rtRootOutputs , 0 } , { rtBlockParameters , 19 , rtModelParameters , 0 } , { ( NULL ) , 0 } , { rtDataTypeMap , rtDimensionMap , rtFixPtMap , rtElementMap , rtSampleTimeMap , rtDimensionArray } , "float" , { 704014149U , 2194897708U , 2911416811U , 3195465174U } , ( NULL ) , 0 , ( boolean_T ) 0 , rt_LoggedStateIdxList } ; const rtwCAPI_ModelMappingStaticInfo * class_d_GetCAPIStaticMap ( void ) { return & mmiStatic ; }
#ifndef HOST_CAPI_BUILD
void class_d_InitializeDataMapInfo ( void ) { rtwCAPI_SetVersion ( ( *
rt_dataMapInfoPtr ) . mmi , 1 ) ; rtwCAPI_SetStaticMap ( ( *
rt_dataMapInfoPtr ) . mmi , & mmiStatic ) ; rtwCAPI_SetLoggingStaticMap ( ( *
rt_dataMapInfoPtr ) . mmi , ( NULL ) ) ; rtwCAPI_SetDataAddressMap ( ( *
rt_dataMapInfoPtr ) . mmi , rtDataAddrMap ) ; rtwCAPI_SetVarDimsAddressMap ( ( *
rt_dataMapInfoPtr ) . mmi , rtVarDimsAddrMap ) ;
rtwCAPI_SetInstanceLoggingInfo ( ( * rt_dataMapInfoPtr ) . mmi , ( NULL ) ) ;
rtwCAPI_SetChildMMIArray ( ( * rt_dataMapInfoPtr ) . mmi , ( NULL ) ) ;
rtwCAPI_SetChildMMIArrayLen ( ( * rt_dataMapInfoPtr ) . mmi , 0 ) ; }
#else
#ifdef __cplusplus
extern "C" {
#endif
void class_d_host_InitializeDataMapInfo ( class_d_host_DataMapInfo_T *
dataMap , const char * path ) { rtwCAPI_SetVersion ( dataMap -> mmi , 1 ) ;
rtwCAPI_SetStaticMap ( dataMap -> mmi , & mmiStatic ) ;
rtwCAPI_SetDataAddressMap ( dataMap -> mmi , ( NULL ) ) ;
rtwCAPI_SetVarDimsAddressMap ( dataMap -> mmi , ( NULL ) ) ; rtwCAPI_SetPath
( dataMap -> mmi , path ) ; rtwCAPI_SetFullPath ( dataMap -> mmi , ( NULL ) )
; rtwCAPI_SetChildMMIArray ( dataMap -> mmi , ( NULL ) ) ;
rtwCAPI_SetChildMMIArrayLen ( dataMap -> mmi , 0 ) ; }
#ifdef __cplusplus
}
#endif
#endif
