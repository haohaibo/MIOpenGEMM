
/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#ifndef GUARD_MIOPENGEMM_OUTPUTWRITER_H
#define GUARD_MIOPENGEMM_OUTPUTWRITER_H

#include <fstream>
#include <iostream>
#include <string>

namespace MIOpenGEMM
{
namespace outputwriting
{

class Flusher
{
  public:
  void increment(){};
};

class Endline
{
  public:
  void increment(){};
};

class OutputWriter
{

  public:
  bool          to_terminal;
  bool          to_file;
  std::ofstream file;
  std::string   filename;

  OutputWriter();
  ~OutputWriter();
  OutputWriter(bool to_terminal, bool to_file, std::string filename = "");
  void operator()(std::string);

  template <typename T>
  OutputWriter& operator<<(T t)
  {

    if (to_terminal)
    {
      std::cout << t;
    }

    if (to_file)
    {
      file << t;
    }
    return *this;
  }
};

template <>
OutputWriter& OutputWriter::operator<<(Flusher f);

template <>
OutputWriter& OutputWriter::operator<<(Endline e);
}

extern outputwriting::Flusher Flush;
extern outputwriting::Endline Endl;
}

#endif
