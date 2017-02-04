#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <sstream>
#include <tinygemm/tinygemmgeometry.hpp>
#include <tinygemm/consistencychecks.hpp>
#include <tinygemm/tinygemmerror.hpp>

namespace tinygemm{

TinyGemmOffsets::TinyGemmOffsets(unsigned oa_, unsigned ob_, unsigned oc_, unsigned oworkspace_):oa(oa_), ob(ob_), oc(oc_), oworkspace(oworkspace_) {}


void TinyGemmGeometryDerived::reset(bool tC, bool isColMajor, unsigned n, unsigned m, char floattype){
  dim_c_coal = tC == isColMajor ? n : m;
  dim_c_uncoal = tC == isColMajor ? m : n;
  if (floattype == 'f'){
    float_size_bytes = sizeof(float);
  }
  else if (floattype == 'd'){
    float_size_bytes = sizeof(double);
  }
  float_size_bits = 8*float_size_bytes;
}


TinyGemmGeometry::TinyGemmGeometry(bool isColMajor_, bool tA_, bool tB_, bool tC_, unsigned lda_, unsigned ldb_, unsigned ldc_, unsigned m_, unsigned n_, unsigned k_, unsigned workspace_size_, char floattype_): isColMajor(isColMajor_), tA(tA_), tB(tB_), tC(tC_), lda(lda_), ldb(ldb_), ldc(ldc_), m(m_), n(n_), k(k_), workspace_size(workspace_size_), floattype(floattype_) {

  
  if (floattype != 'd' and floattype != 'f'){
    throw tinygemm::tinygemm_error("floattype should be one of 'f' and 'd' (in TinyGemmGeometry constructor)");
  }
    
  consistencychecks::check_ldx_mnk_consistent(isColMajor,  tA,  tB,  tC,  lda,  ldb,  ldc,  m,  n,  k); //, a_offset, b_offset, c_offset
  
  derived.reset(tC, isColMajor, n, m, floattype);

}

std::string TinyGemmGeometry::get_string() const{
  std::stringstream geometry_stringstream;
  geometry_stringstream << " tC:" << tC << " tA:" << tA << " tB:" << tB << " colMaj:" << isColMajor << " m:" << m << " n:" << n << " k:" << k << " lda:" << lda << " ldb:" << ldb << " ldc:" << ldc  << " workspace_size:" << workspace_size << " floattype:" << floattype;
  return geometry_stringstream.str();
}
// " a_offset:" << a_offset << " b_offset:" << b_offset << " c_offset:" << c_offset << " workspace_offset:" << workspace_offset


std::string TinyGemmGeometry::get_networkconfig_string() const{
  std::stringstream geometry_stringstream;
  geometry_stringstream << "tC" << tC << "_tA" << tA << "_tB" << tB << "_colMaj" << isColMajor << "_m" << m << "_n" << n << "_k" << k << "_lda" << lda << "_ldb" << ldb << "_ldc" << ldc << "_ws" << workspace_size << "_f" << derived.float_size_bits;
  return geometry_stringstream.str();
}



float TinyGemmGeometry::get_distance(const TinyGemmGeometry & gg) const{
  /* problems which are "larger" are infinitely far away (as their tile might not fit) */
  
  
  
  float distance;
  
  if (workspace_size < gg.workspace_size || floattype != gg.floattype || tA != gg.tA || tB != gg.tB || isColMajor != gg.isColMajor || (m < std::min<unsigned>(600, gg.m)) || n < std::min<unsigned>(600, gg.n)){
    distance = std::numeric_limits<float>::max();
  } 
   
  else{
    distance =  
    0.2*std::abs(float(k) - float(gg.k)) + 
    1.0*std::abs(float(m) - float(gg.m)) + 
    1.0*std::abs(float(n) - float(gg.n)) + 
    1.0*std::abs(float(lda) - float(gg.lda)) + 
    1.0*std::abs(float(ldb) - float(gg.ldb)) + 
    0.2*std::abs(float(ldc) - float(gg.ldc));
  }
  

  
  
  return distance;
}

}
