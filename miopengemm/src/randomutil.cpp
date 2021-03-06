/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#include <miopengemm/randomutil.hpp>

namespace MIOpenGEMM
{

RandomUtil::RandomUtil() : rd(), gen(rd()) {}

unsigned RandomUtil::get_from_range(unsigned upper) { return unidis(gen) % upper; }
}
