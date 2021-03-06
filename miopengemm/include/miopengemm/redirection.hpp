/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_REDIRECTION_HPP
#define GUARD_MIOPENGEMM_REDIRECTION_HPP

namespace MIOpenGEMM
{
namespace redirection
{

template <typename TFloat>
void redirect(bool&          isColMajor,
              bool&          tA,
              bool&          tB,
              bool&          tC,
              unsigned&      m,
              unsigned&      n,
              unsigned&      lda,
              unsigned&      ldb,
              unsigned&      a_offset,
              unsigned&      b_offset,
              const TFloat*& a,
              const TFloat*& b);

void redirect(bool&        isColMajor,
              bool&        tA,
              bool&        tB,
              bool&        tC,
              unsigned&    m,
              unsigned&    n,
              std::string& a,
              std::string& b);

void confirm_redirection(bool isColMajor, bool tA, bool tB, unsigned m, unsigned n);

}
}
#endif
