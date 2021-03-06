/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_PROBLEMGEOMETRY_UTIL_HPP
#define GUARD_MIOPENGEMM_PROBLEMGEOMETRY_UTIL_HPP

#include <string>
#include <vector>
#include <miopengemm/geometry.hpp>

namespace MIOpenGEMM
{

std::vector<Geometry> get_from_m_n_k_ldaABC_tA_tB(
  const std::vector<
    std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, bool, bool>>& basicgeos,
  unsigned workspace_size);

std::vector<Geometry> get_from_m_n_k_tA_tB(
  const std::vector<std::tuple<unsigned, unsigned, unsigned, bool, bool>>& basicgeos,
  unsigned workspace_size);
}

#endif
