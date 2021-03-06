/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#include <chrono>
#include <iomanip>
#include <miopengemm/error.hpp>
#include <miopengemm/kernel.hpp>
#include <miopengemm/openclutil.hpp>

namespace MIOpenGEMM
{

Kernel::Kernel(cl_command_queue command_queue_, const std::string& hash_)
  : command_queue(command_queue_), clprog(nullptr), clkern(nullptr), hash(hash_)
{
}

void Kernel::try_release()
{

  if (clprog != nullptr)
  {
    openclutil::cl_release_program(clprog, "Kernel Destructor", true);
  }
  if (clkern != nullptr)
  {
    openclutil::cl_release_kernel(clkern, "Kernel Destructor", true);
  }
}

openclutil::OpenCLResult Kernel::update(const KernelString& ks, outputwriting::OutputWriter& mowri)
{

  try_release();
  tgk_strings = ks;
  mowri << "compiling " << ks.type.bkt_string << ". " << Flush;

  auto start = std::chrono::high_resolution_clock::now();

  auto oclr = openclutil::cl_set_program_and_kernel(
    command_queue, tgk_strings.kernstr, tgk_strings.fname, clprog, clkern, mowri, false);

  auto                         end             = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> fp_ms           = end - start;
  float                        elapsed_seconds = fp_ms.count();

  if (oclr.fail())
  {
    mowri << "Failed in " << std::setprecision(3) << elapsed_seconds << std::setprecision(6)
          << " [s]" << Endl;
  }
  else
  {
    mowri << "Done in " << std::setprecision(3) << elapsed_seconds << std::setprecision(6) << " [s]"
          << Endl;
  }
  return oclr;
}

Kernel::~Kernel() { try_release(); }

bool Kernel::is_set() { return (clprog != nullptr && clkern != nullptr); }

void Kernel::set_kernel_arg(cl_uint arg_index, size_t arg_size, const void* arg_value)
{

  if (clkern == nullptr)
  {
    throw miog_error("Attempt to set kernel argument of uninitialised kernel");
  }

  openclutil::cl_set_kernel_arg(clkern,
                                arg_index,
                                arg_size,
                                arg_value,
                                "in set_kernel_arg of Kernel, " + hash + " index : " +
                                  std::to_string(arg_index),
                                true);
}

void Kernel::set_kernel_args(std::vector<std::pair<size_t, const void*>> arg_sizes_values)
{
  for (cl_uint arg_index = 0; arg_index < arg_sizes_values.size(); ++arg_index)
  {
    set_kernel_arg(
      arg_index, arg_sizes_values[arg_index].first, arg_sizes_values[arg_index].second);
  }
}

openclutil::OpenCLResult Kernel::enqueue(cl_uint         num_events_in_wait_list,
                                         const cl_event* event_wait_list)
{

  return openclutil::cl_enqueue_ndrange_kernel(command_queue,
                                               clkern,
                                               1,
                                               NULL,
                                               &tgk_strings.global_work_size,
                                               &tgk_strings.local_work_size,
                                               num_events_in_wait_list,
                                               event_wait_list,
                                               &clevent,
                                               "Kernel::enqueue",
                                               false);
}

openclutil::OpenCLResult Kernel::enqueue() { return enqueue(0, nullptr); }

void Kernel::update_times()
{

  openclutil::cl_set_event_profiling_info(clevent,
                                          CL_PROFILING_COMMAND_START,
                                          sizeof(size_t),
                                          &t_start,
                                          nullptr,
                                          "in update_times",
                                          true);
  openclutil::cl_set_event_profiling_info(
    clevent, CL_PROFILING_COMMAND_END, sizeof(size_t), &t_end, nullptr, "in update_times", true);
  v_times.push_back(1e-6 * (t_end - t_start));
}

void Kernel::reset_times()
{
  t_start = 0;
  t_end   = 0;
  v_times.resize(0);
}
}
