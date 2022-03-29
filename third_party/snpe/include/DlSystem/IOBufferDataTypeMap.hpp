//=============================================================================
//
//  Copyright (c) 2021 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================


#ifndef DL_SYSTEM_IOBUFFER_DATATYPE_MAP_HPP
#define DL_SYSTEM_IOBUFFER_DATATYPE_MAP_HPP

#include <initializer_list>
#include <cstdio>
#include <memory>
#include "DlSystem/DlEnums.hpp"

namespace DlSystem
{
   // Forward declaration of IOBufferDataTypeMapImpl implementation.
   class IOBufferDataTypeMapImpl;
}

namespace zdl
{
namespace DlSystem
{
class ZDL_EXPORT IOBufferDataTypeMap final
{
public:

    /**
    * @brief .
    *
    * Creates a new Buffer Data type map
    *
    */
   IOBufferDataTypeMap();

   /**
    * @brief Adds a name and the corresponding buffer data type
    *        to the map
    *
    * @param[name] name The name of the buffer
    * @param[bufferDataType] buffer Data Type of the buffer
    *
    * @note If a buffer with the same name already exists, no new
    *       buffer is added.
    */
   void add(const char* name, zdl::DlSystem::IOBufferDataType_t bufferDataType);

   void remove(const char* name);

   zdl::DlSystem::IOBufferDataType_t getBufferDataType(const char* name);

   zdl::DlSystem::IOBufferDataType_t getBufferDataType();

   size_t size();

   bool find(const char* name);

   void clear();

   bool empty();

   ~IOBufferDataTypeMap();

private:
   std::shared_ptr<::DlSystem::IOBufferDataTypeMapImpl> m_IOBufferDataTypeMapImpl;
};
}

}
#endif