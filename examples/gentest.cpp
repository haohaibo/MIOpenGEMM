/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved.
 *******************************************************************************/
#include <string>
#include <miopengemm/standalone.hpp>

int main()
{
  using namespace MIOpenGEMM;



  std::vector<std::pair<Geometry, HyPas>> incorrect = 
  {
  //0
  {{"tC1_tA1_tB1_colMaj1_m45_n56_k64_lda64_ldb64_ldc64_ws1_f32"}, 
  {"A_MIC2_PAD2_PLU0_LIW0_MIW1_WOS0_VEW1__B_MIC6_PAD1_PLU0_LIW0_MIW1_WOS0_VEW1__C_UNR16_GAL3_PUN1_ICE2_IWI0_SZT0_NAW16_UFO0_MAC1_SKW10_AFI1_MIA0"}},
  //1
  {{"tC0_tA1_tB0_colMaj0_m2048_n121_k1_lda2048_ldb121_ldc121_ws0_f32"},
  {"A_MIC1_PAD0_PLU0_LIW0_MIW1_WOS0_VEW1__B_MIC1_PAD0_PLU1_LIW0_MIW0_WOS0_VEW1__C_UNR32_GAL2_PUN1_ICE1_IWI1_SZT0_NAW16_UFO0_MAC64_SKW10_AFI1_MIA0"}},
  //2
  {{"tC0_tA1_tB0_colMaj0_m1601_n64_k1_lda1601_ldb269_ldc269_ws1_f32"}, 
  {"A_MIC2_PAD0_PLU0_LIW0_MIW0_WOS0_VEW1__B_MIC4_PAD0_PLU1_LIW0_MIW0_WOS0_VEW1__C_UNR16_GAL1_PUN0_ICE1_NAW64_IWI1_SZT0_UFO0_MAC256_SKW9_AFI1_MIA0"}}
  };

  
  
  std::vector<std::pair<Geometry, HyPas>> freeze = 
  {
  //0
  {{"tC0_tA0_tB0_colMaj1_m2560_n65_k2560_lda2560_ldb2560_ldc2560_ws0_f32"}, 
  {"A_MIC6_PAD2_PLU1_LIW0_MIW0_WOS0_VEW1__B_MIC6_PAD1_PLU0_LIW1_MIW0_WOS0_VEW1__C_UNR16_GAL3_PUN1_ICE1_IWI1_SZT0_NAW64_UFO0_MAC16_SKW8_AFI1_MIA0"}}, 
  //1
  {{"tC0_tA0_tB0_colMaj1_m512_n8_k500000_lda512_ldb500000_ldc512_ws0_f32"},
  {"A_MIC6_PAD0_PLU1_LIW0_MIW0_WOS0_VEW1__B_MIC3_PAD2_PLU1_LIW1_MIW1_WOS0_VEW1__C_UNR32_GAL3_PUN1_ICE8_IWI0_SZT0_NAW16_UFO0_MAC16_SKW8_AFI1_MIA0"}},
  //2
  {{"tC0_tA0_tB0_colMaj1_m512_n8_k500000_lda512_ldb500000_ldc512_ws0_f32"},
  {"A_MIC6_PAD2_PLU1_LIW0_MIW0_WOS0_VEW1__B_MIC6_PAD0_PLU0_LIW0_MIW1_WOS0_VEW1__C_UNR16_GAL2_PUN1_ICE2_IWI1_SZT0_NAW16_UFO1_MAC4_SKW9_AFI1_MIA0"}},
  //3
  {{"tC0_tA1_tB0_colMaj1_m363_n1_k576_lda576_ldb576_ldc363_ws0_f32"},
  {"A_MIC3_PAD1_PLU1_LIW0_MIW1_WOS0_VEW1__B_MIC1_PAD2_PLU1_LIW0_MIW1_WOS0_VEW1__C_UNR64_GAL2_PUN1_ICE1_IWI0_SZT0_NAW64_UFO0_MAC64_SKW7_AFI1_MIA0"}},
  //4
  {{"tC0_tA0_tB0_colMaj1_m25_n5_k25_lda25_ldb25_ldc25_ws0_f32"},
  {"A_MIC1_PAD0_PLU1_LIW1_MIW1_WOS0_VEW1__B_MIC1_PAD0_PLU1_LIW1_MIW0_WOS0_VEW1__C_UNR16_GAL2_PUN1_ICE1_IWI1_SZT0_NAW64_UFO0_MAC4_SKW11_AFI1_MIA0"}},
  //5
  {{"tC0_tA0_tB0_colMaj1_m77_n1002_k77_lda77_ldb77_ldc77_ws0_f32"},
  {"A_MIC1_PAD2_PLU0_LIW1_MIW1_WOS0_VEW1__B_MIC8_PAD1_PLU1_LIW0_MIW1_WOS0_VEW1__C_UNR64_GAL3_PUN0_ICE1_IWI1_SZT0_NAW16_UFO0_MAC64_SKW10_AFI1_MIA0"}},
  //6
  {{"tC0_tA0_tB0_colMaj1_m63_n63_k63_lda63_ldb63_ldc63_ws0_f32"}, 
  {"A_MIC6_PAD2_PLU0_LIW0_MIW1_WOS0_VEW1__B_MIC4_PAD1_PLU1_LIW0_MIW1_WOS0_VEW1__C_UNR16_GAL3_PUN1_ICE1_IWI1_SZT0_NAW64_UFO0_MAC16_SKW10_AFI1_MIA0"}},
  //7
  {{"tC0_tA0_tB0_colMaj1_m252_n252_k252_lda252_ldb252_ldc252_ws0_f32"},
  {"A_MIC2_PAD1_PLU0_LIW0_MIW0_WOS0_VEW2__B_MIC2_PAD2_PLU0_LIW0_MIW1_WOS0_VEW1__C_UNR64_GAL3_PUN0_ICE1_IWI0_SZT0_NAW16_UFO0_MAC64_SKW10_AFI1_MIA0"}},
  //8
  {{"tC0_tA0_tB0_colMaj1_m36_n36_k36_lda36_ldb36_ldc36_ws0_f32"},
  {"A_MIC2_PAD1_PLU0_LIW0_MIW0_WOS0_VEW1__B_MIC2_PAD2_PLU1_LIW1_MIW0_WOS0_VEW1__C_UNR16_GAL1_PUN0_ICE1_IWI1_SZT0_NAW16_UFO0_MAC16_SKW11_AFI1_MIA0"}},
  //9
  {{"tC0_tA0_tB1_colMaj1_m127_n127_k127_lda127_ldb127_ldc127_ws0_f32"},
  {"A_MIC1_PAD2_PLU0_LIW0_MIW0_WOS0_VEW1__B_MIC4_PAD1_PLU0_LIW1_MIW0_WOS0_VEW2__C_UNR64_GAL3_PUN1_ICE1_IWI0_SZT0_NAW16_UFO0_MAC64_SKW9_AFI1_MIA0"}},
  };


  owrite::Writer mowri(Ver::E::TERMINAL, "");

  

  // hangs :  
  Geometry gg("tC0_tA0_tB1_colMaj1_m127_n127_k127_lda127_ldb127_ldc127_ws0_f32");
  HyPas hp("A_MIC1_PAD2_PLU0_LIW0_MIW0_WOS0_VEW1__B_MIC4_PAD1_PLU0_LIW1_MIW0_WOS0_VEW2__C_UNR64_GAL3_PUN1_ICE1_IWI0_SZT0_NAW16_UFO0_MAC64_SKW9_AFI1_MIA0");
  auto standalone_source = standalone::make(gg, hp, mowri);
  auto fname = "/home/james/MIOpenGEMM/tests/hangs1.cpp";
  mowri << "writing " << fname << " ... " << Flush;
  std::ofstream floper(fname, std::ios::out);
  floper << standalone_source;
  floper.close();
  mowri << "done." << Endl;


  // incorrect : 
  gg = Geometry("tC0_tA1_tB0_colMaj0_m2048_n121_k1_lda2048_ldb121_ldc121_ws0_f32");
  hp = HyPas("A_MIC1_PAD0_PLU0_LIW0_MIW1_WOS0_VEW1__B_MIC1_PAD0_PLU1_LIW0_MIW0_WOS0_VEW1__C_UNR32_GAL2_PUN0_ICE1_IWI1_SZT0_NAW16_UFO0_MAC64_SKW10_AFI1_MIA0");
  standalone_source = standalone::make(gg, hp, mowri);
  fname = "/home/james/MIOpenGEMM/tests/incorrect1.cpp";
  mowri << "writing " << fname << " ... " << Flush;
  floper = std::ofstream(fname, std::ios::out);
  floper << standalone_source;
  floper.close();
  mowri << "done." << Endl;

  // incorrect : 
  gg = Geometry("tC0_tA0_tB0_colMaj1_m256_n256_k256_lda256_ldb256_ldc256_ws0_f32");
  hp = HyPas("A_MIC4_PAD1_PLU0_LIW0_MIW1_WOS0_VEW1__B_MIC4_PAD1_PLU1_LIW0_MIW0_WOS0_VEW1__C_UNR16_GAL1_PUN0_ICE1_IWI1_SZT0_NAW16_UFO0_MAC64_SKW10_AFI1_MIA0");
  standalone_source = standalone::make(gg, hp, mowri);
  fname = "/home/james/MIOpenGEMM/tests/epsilonoff1.cpp";
  mowri << "writing " << fname << " ... " << Flush;
  floper = std::ofstream(fname, std::ios::out);
  floper << standalone_source;
  floper.close();
  mowri << "done." << Endl;


  

  return 0;
}