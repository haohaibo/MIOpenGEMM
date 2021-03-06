/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#include <sstream>
#include <miopengemm/error.hpp>
#include <miopengemm/setabcw.hpp>

namespace MIOpenGEMM
{
namespace setabcw
{

template <typename TFloat>
void fill_uni(std::vector<TFloat>& v, unsigned r_small, unsigned r_big)
{

  if (r_small > r_big)
  {
    std::stringstream ss;
    ss << "strange request : in fill_uni, with r_small > r_big";
    throw miog_error(ss.str());
  }

  if (r_small > v.size())
  {
    throw miog_error("strange request : in fill_uni, r_small > v.size()");
  }

  if (r_big > v.size())
  {
    throw miog_error("strange request : in fill_uni, r_big > v.size()");
  }

  for (size_t i = 0; i < r_small; ++i)
  {
    v[i] = TFloat(rand() % 1000) / 1000.;
  }

  for (size_t i = r_small; i < r_big; ++i)
  {
    v[i] = 1e9 * (TFloat(rand() % 1000) / 1000.);
  }
}

void set_nelmts_abc(const Geometry& gg, const Offsets& toff, size_t & n_a, size_t & n_b, size_t & n_c){
  n_a = gg.ldX[nsHP::matA] * (gg.tX[nsHP::matA] == gg.isColMajor ? gg.m : gg.k) + toff.oa + toff.tail_off_a;
  n_b = gg.ldX[nsHP::matB] * (gg.tX[nsHP::matB] == gg.isColMajor ? gg.k : gg.n) + toff.ob + toff.tail_off_b;
  n_c = gg.ldX[nsHP::matC] * (gg.tX[nsHP::matC] == gg.isColMajor ? gg.m : gg.n) + toff.oc + toff.tail_off_c;
}


  //size_t n_a, n_b, n_c;
  //set_nelmts_abc(gg, toff, n_a, n_b, n_c);
  //size_t n_elmnts_limit = 20000 * 10000;
  //if (n_a > n_elmnts_limit || n_b > n_elmnts_limit || n_c > n_elmnts_limit)
  //{
    //std::stringstream ss;
    //ss << "currently, this code only generates random matrices of size less "
       //<< "than " << n_elmnts_limit << " elements. The request here is for n_a=" << n_a << " n_b=" << n_b << " n_c= " << n_c;
    //throw miog_error(ss.str());
  //}
  //// fill matrices with random floats.
  //// Sometimes it seems to be important
  //// to fill them with random floats,
  //// as if they're integers, the kernel
  //// can sometimes cheat! (runs faster)
  //v_a.resize(n_a);
  //v_b.resize(n_b);
  //v_c.resize(n_c);
  
  //fill_uni<TFloat>(v_a, n_a - toff.tail_off_a, n_a);
  //fill_uni<TFloat>(v_b, n_b - toff.tail_off_b, n_b);
  //fill_uni<TFloat>(v_c, n_c - toff.tail_off_c, n_c);
//}


template <typename TFloat>
void set_multigeom_abc(std::vector<TFloat>& v_a,
             std::vector<TFloat>& v_b,
             std::vector<TFloat>& v_c,
             const std::vector<Geometry>& ggs,
             const Offsets&       toff)
{

  size_t n_a{0};
  size_t n_b{0};
  size_t n_c{0};
  size_t tn_a, tn_b, tn_c;
  
  for (auto & gg : ggs){
    set_nelmts_abc(gg, toff, tn_a, tn_b, tn_c);
    n_a = std::max(n_a, tn_a);
    n_b = std::max(n_b, tn_b);
    n_c = std::max(n_c, tn_c);       
  }
  
  size_t n_elmnts_limit = 20000 * 10000;
  if (n_a > n_elmnts_limit || n_b > n_elmnts_limit || n_c > n_elmnts_limit)
  {
    std::stringstream ss;
    ss << "currently, this code only generates random matrices of size less "
       << "than " << n_elmnts_limit << " elements. The request here is for n_a=" << n_a << " n_b=" << n_b << " n_c= " << n_c;
    throw miog_error(ss.str());
  }
  // fill matrices with random floats.
  // Sometimes it seems to be important
  // to fill them with random floats,
  // as if they're integers, the kernel
  // can sometimes cheat! (runs faster)
  v_a.resize(n_a);
  v_b.resize(n_b);
  v_c.resize(n_c);
  
  fill_uni<TFloat>(v_a, n_a - toff.tail_off_a, n_a);
  fill_uni<TFloat>(v_b, n_b - toff.tail_off_b, n_b);
  fill_uni<TFloat>(v_c, n_c - toff.tail_off_c, n_c);
  
}


template <typename TFloat>
void set_abc(std::vector<TFloat>& v_a,
             std::vector<TFloat>& v_b,
             std::vector<TFloat>& v_c,
             const Geometry&      gg,
             const Offsets&       toff)
{
  set_multigeom_abc(v_a, v_b, v_c, {gg}, toff);
}


template <typename TFloat>
void set_abcw(std::vector<TFloat>& v_a,
              std::vector<TFloat>& v_b,
              std::vector<TFloat>& v_c,
              std::vector<TFloat>& v_workspace,
              const Geometry&      gg,
              const Offsets&       toff)
{

  set_abc<TFloat>(v_a, v_b, v_c, gg, toff);

  size_t n_workspace = gg.workspace_size + toff.oworkspace;

  v_workspace.resize(n_workspace);
  fill_uni(v_workspace, n_workspace, n_workspace);
}

template void set_abc(std::vector<double>& v_a,
                      std::vector<double>& v_b,
                      std::vector<double>& v_c,
                      const Geometry&      gg,
                      const Offsets&       toff);

template void set_abc(std::vector<float>& v_a,
                      std::vector<float>& v_b,
                      std::vector<float>& v_c,
                      const Geometry&     gg,
                      const Offsets&      toff);

template void set_multigeom_abc(std::vector<double>& v_a,
                      std::vector<double>& v_b,
                      std::vector<double>& v_c,
                      const std::vector<Geometry>&,
                      const Offsets&       toff);

template void set_multigeom_abc(std::vector<float>& v_a,
                      std::vector<float>& v_b,
                      std::vector<float>& v_c,
                      const std::vector<Geometry>&,
                      const Offsets&      toff);
                      
template void set_abcw(std::vector<double>& v_a,
                       std::vector<double>& v_b,
                       std::vector<double>& v_c,
                       std::vector<double>& v_workspace,
                       const Geometry&      gg,
                       const Offsets&       toff);

template void set_abcw(std::vector<float>& v_a,
                       std::vector<float>& v_b,
                       std::vector<float>& v_c,
                       std::vector<float>& v_workspace,
                       const Geometry&     gg,
                       const Offsets&      toff);
}
}
