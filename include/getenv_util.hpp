/*
// Copyright (c) 2022-2024 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/
#pragma once

#include <stdlib.h>
#include <string.h>

#include <string>

#if defined(_WIN32)

#include <windows.h>

#define GETENV( _name, _value ) _dupenv_s( &_value, NULL, _name )
#define FREEENV( _value ) free( _value )

#else

#define GETENV( _name, _value ) _value = getenv(_name)
#define FREEENV( _value ) (void)_value

#endif

static inline bool getControlFromEnvironment(
    const char* name,
    void* pValue,
    size_t size )
{
    char* envVal = NULL;
    GETENV( name, envVal );

    if( envVal != NULL )
    {
        if( size == sizeof(unsigned int) )
        {
            unsigned int* puVal = (unsigned int*)pValue;
            *puVal = atoi(envVal);
        }
        else if( strlen(envVal) < size )
        {
            char* pStr = (char*)pValue;
            strcpy( pStr, envVal );
        }

        FREEENV( envVal );
        return true;
    }

    return false;
}

template <class T>
static bool getControl(
    const char* name,
    T& value )
{
    unsigned int readValue = 0;
    bool success = getControlFromEnvironment( name, &readValue, sizeof(readValue) );
    if( success )
    {
        value = readValue;
    }

    return success;
}

template <>
bool getControl<bool>(
    const char* name,
    bool& value )
{
    unsigned int readValue = 0;
    bool success = getControlFromEnvironment( name, &readValue, sizeof(readValue) );
    if( success )
    {
        value = ( readValue != 0 );
    }

    return success;
}

template <>
bool getControl<std::string>(
    const char* name,
    std::string& value )
{
    char readValue[256] = "";
    bool success = getControlFromEnvironment( name, readValue, sizeof(readValue) );
    if( success )
    {
        value = readValue;
    }

    return success;
}
