/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_STRINGUTILBASE_HPP
#define GUARD_MIOPENGEMM_STRINGUTILBASE_HPP

#include <string>
#include <vector>

namespace MIOpenGEMM
{
namespace stringutil
{

void indentify(std::string& source);

// split the string tosplit by delim.
// With x appearances of delim in tosplit,
// the returned vector will have length x + 1
// (even if appearances at the start, end, contiguous.
std::vector<std::string> split(const std::string& tosplit, const std::string& delim);

// split on whitespaces
std::vector<std::string> split(const std::string& tosplit);

std::string getdirfromfn(const std::string& fn);

// split something like QWE111 into QWE and 111.
std::tuple<std::string, unsigned> splitnumeric(std::string alphanum);

std::string get_padded(unsigned x, unsigned length = 4);

template <typename T>
std::string get_char_padded(const T& t, unsigned length){
  auto t_s = std::to_string(t);
  if (t_s.size() < length)
  t_s.resize(length, ' ');
  return t_s;
}


}
}

#endif
