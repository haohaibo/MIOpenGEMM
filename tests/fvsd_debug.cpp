#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>

#include <tinygemm/tinygemmerror.hpp>
#include <tinygemm/tinygemm.hpp>
#include <tinygemm/devtinygemm.hpp>
#include <tinygemm/bundle.hpp>
#include <tinygemm/stringutilbase.hpp>
#include <tinygemm/hyperparams.hpp>
#include <tinygemm/tinygemmgeometry.hpp>
#include <tinygemm/outputwriter.hpp>

#include <tinygemm/setabcw.hpp>


std::vector<std::tuple<std::string, std::string>> get_tests(char ftype){
  
  
  std::string nbits = ftype == 'f' ? "32" : "64";
   
  std::vector<std::tuple<std::string, std::string>> tests = {

   std::make_tuple(
   "A_MIC2_PAD0_PLU0_LIW0_MIW0_WOS0__B_MIC4_PAD0_PLU1_LIW0_MIW0_WOS0__C_UNR16_GAL1_PUN0_ICE1_NAW64_UFO0_MAC256_SKW9", 
   "tC0_tA1_tB0_colMaj0_m1601_n64_k1_lda1601_ldb269_ldc269_ws0_f" + nbits),
   
   /* perturb n by 1 */
   std::make_tuple(
   "A_MIC2_PAD0_PLU0_LIW0_MIW0_WOS0__B_MIC4_PAD0_PLU1_LIW0_MIW0_WOS0__C_UNR16_GAL1_PUN0_ICE1_NAW64_UFO0_MAC256_SKW9", 
   "tC0_tA1_tB0_colMaj0_m1601_n65_k1_lda1601_ldb269_ldc269_ws0_f" + nbits),

   /* perturb m by 1 */
   std::make_tuple(
   "A_MIC2_PAD0_PLU0_LIW0_MIW0_WOS0__B_MIC4_PAD0_PLU1_LIW0_MIW0_WOS0__C_UNR16_GAL1_PUN0_ICE1_NAW64_UFO0_MAC256_SKW9", 
   "tC0_tA1_tB0_colMaj0_m1600_n64_k1_lda1601_ldb269_ldc269_ws0_f" + nbits),

   /* perturb k by 1 */
   std::make_tuple(
   "A_MIC2_PAD0_PLU0_LIW0_MIW0_WOS0__B_MIC4_PAD0_PLU1_LIW0_MIW0_WOS0__C_UNR16_GAL1_PUN0_ICE1_NAW64_UFO0_MAC256_SKW9", 
   "tC0_tA1_tB0_colMaj0_m1601_n64_k2_lda1601_ldb269_ldc269_ws0_f" + nbits),

   /* perturb C_SKW by 1 */
   std::make_tuple(
   "A_MIC2_PAD0_PLU0_LIW0_MIW0_WOS0__B_MIC4_PAD0_PLU1_LIW0_MIW0_WOS0__C_UNR16_GAL1_PUN0_ICE1_NAW64_UFO0_MAC256_SKW10", 
   "tC0_tA1_tB0_colMaj0_m1601_n64_k1_lda1601_ldb269_ldc269_ws0_f" + nbits),
   
    /* perturb B_PLU by 1 */
    std::make_tuple(
   "A_MIC2_PAD0_PLU0_LIW0_MIW0_WOS0__B_MIC4_PAD0_PLU0_LIW0_MIW0_WOS0__C_UNR16_GAL1_PUN0_ICE1_NAW64_UFO0_MAC256_SKW9", 
   "tC0_tA1_tB0_colMaj0_m1601_n64_k1_lda1601_ldb269_ldc269_ws0_f" + nbits),
   
  };
  
  return tests;
}


tinygemm::TinyGemmOffsets get_offsets(){

  unsigned a_offset = 0;//330;
  unsigned b_offset = 0;//550;
  unsigned c_offset = 0;//770;
  unsigned workspace_offset = 0;
  unsigned tail_off_a = 0;//1e6 + 123;
  unsigned tail_off_b = 0;//1e6 + 97;
  unsigned tail_off_c = 0;//1e6 + 67;
  return {a_offset, b_offset, c_offset, workspace_offset, tail_off_a, tail_off_b, tail_off_c};

}


template <typename TFloat>
void print_kernel(std::string hyperstring, std::string ggs){
  std::string kernel_string;  
  tinygemm::TinyGemmGeometry gg(ggs);;
  tinygemm::openclutil::OpenCLDeviceInfo devinfo;
  devinfo.wg_atom_size = 32;
  tinygemm::hyperparams::Graph graph(gg, devinfo, hyperstring, true); 
  tinygemm::hyperparams::HyperParams hp(graph);
  bool mowri_verbose = true;
  bool verbose_get_bundle = true;
  std::string mowri_out("");
  tinygemm::outputwriting::OutputWriter mowri(mowri_verbose, mowri_out != "" , mowri_out);
  auto bundle = tinygemm::kerngen::get_bundle(hp, gg, mowri, verbose_get_bundle);
  for (auto & x :  bundle.v_tgks){
    auto dirname = "/home/james/01/" + gg.get_string() + "/" + hyperstring + "/";
    std::string syscall = "mkdir -p " + dirname;
    std::system(syscall.c_str());
    auto fname =  dirname +  x.type.full +  ".cl";
    std::cout << "writing " << fname << " ... " << std::flush;
    std::ofstream floper (fname, std::ios::out); 
    floper << x.kernstr;
    floper.close();
    std::cout << "done." << std::endl;
  }
}
  



template <typename TFloat>
int test_loop(){

  std::string fout("");
  tinygemm::outputwriting::OutputWriter mowri(true, fout != "" , fout);
  
  char ft = sizeof(TFloat) == 4 ? 'f' : 'd';
  for (auto & test : get_tests(ft)){
  
    auto hyperstring = std::get<0>(test);
    auto ggstring = std::get<1>(test);

    mowri << "\n\n" << hyperstring << "\n" << ggstring << "\n"    ;
    
    tinygemm::TinyGemmGeometry gg(ggstring);
    tinygemm::TinyGemmOffsets toff = get_offsets();  
      
    std::vector<TFloat> v_a;
    std::vector<TFloat> v_b;
    std::vector<TFloat> v_c;
    setabcw::set_abc<TFloat>(v_a, v_b, v_c, gg, toff);
    const TFloat * c_true_bla = nullptr; 
   
    //if (test_print){
      //print_kernel<tfloat>();
    //}
    
  
    tinygemm::dev::accuracy_test(hyperstring, gg, toff, v_a.data(), v_b.data(), v_c.data(), c_true_bla, mowri);
      
  }
      
 
  return 0;
}

int main(){
  
  std::cout << R"(
   ***
  *****
  float
  *****
   ***)";
  test_loop<float>();


  std::cout << R"(
   ****
  ******
  double
  ******
   ****)";

  test_loop<double>();
  return  0;
}

