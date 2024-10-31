#ifndef class_d_cap_host_h__
#define class_d_cap_host_h__
#ifdef HOST_CAPI_BUILD
#include "rtw_capi.h"
#include "rtw_modelmap_simtarget.h"
typedef struct { rtwCAPI_ModelMappingInfo mmi ; } class_d_host_DataMapInfo_T
;
#ifdef __cplusplus
extern "C" {
#endif
void class_d_host_InitializeDataMapInfo ( class_d_host_DataMapInfo_T *
dataMap , const char * path ) ;
#ifdef __cplusplus
}
#endif
#endif
#endif
