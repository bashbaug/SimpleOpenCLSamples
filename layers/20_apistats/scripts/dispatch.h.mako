<%
%>/*
// Copyright (c) 2023 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <CL/cl.h>

#include "scope_profiler.h"

extern const struct _cl_icd_dispatch* g_pNextDispatch;
%for function in spec.findall('feature/require/command'):
<%
      api = apisigs[function.get('name')]
%>
///////////////////////////////////////////////////////////////////////////////

static ${api.RetType} CL_API_CALL ${api.Name}_layer(
%if len(api.Params) == 0:
    void)
%else:
%  for i, param in enumerate(api.Params):
%    if i < len(api.Params)-1:
    ${param.Type} ${param.Name}${param.TypeEnd},
%    else:
    ${param.Type} ${param.Name}${param.TypeEnd})
%    endif
%  endfor
%endif
{
    PROFILE_SCOPE("${api.Name}");
%if len(api.Params) == 0:
    return g_pNextDispatch->${api.Name}();
%else:
    return g_pNextDispatch->${api.Name}(
%  for i, param in enumerate(api.Params):
%    if i < len(api.Params)-1:
        ${param.Name},
%    else:
        ${param.Name});
%    endif
%  endfor
%endif
}
%endfor
