#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>

#include <MIOpenGEMM/error.hpp>
#include <MIOpenGEMM/miogemm.hpp>
#include <MIOpenGEMM/devmiogemm.hpp>
#include <MIOpenGEMM/bundle.hpp>
#include <MIOpenGEMM/stringutilbase.hpp>
#include <MIOpenGEMM/hyperparams.hpp>
#include <MIOpenGEMM/geometry.hpp>
#include <MIOpenGEMM/outputwriter.hpp>

#include <MIOpenGEMM/setabcw.hpp>
/* Note that currently (13/11/2016) most testing is done through dev/python scripts */
/* Update (30/01/2017) this is the preferred place for doing testing */


std::string get_hyperstring(std::string hyperstring = ""){
  if (hyperstring.compare("") == 0){
    hyperstring = "A_MIC2_PAD0_PLU0_LIW0_MIW0_WOS0__B_MIC4_PAD0_PLU1_LIW0_MIW0_WOS0__C_UNR16_GAL1_PUN0_ICE1_NAW64_UFO0_MAC256_SKW9";
  }
  return hyperstring;
}

template <typename TFloat>
MIOpenGEMM::Geometry get_geometry(){


  //bool isColMajor = true;
  //bool tA = false;
  //bool tB = false;
  //bool tC = false;
  //unsigned m = 512; 
  //unsigned n = 512; 
  //unsigned k = 512;             
  //unsigned lda = ( tA == isColMajor ? k : m ) + 0;
  //unsigned ldb = ( tB == isColMajor ? n : k ) + 0;
  //unsigned ldc = ( tC == isColMajor ? n : m ) + 0;
  //unsigned workspace_size =  0;
  //char floattype = sizeof(TFloat) == sizeof(double) ? 'd' : 'f';
  //return { isColMajor, tA, tB, tC, lda, ldb, ldc, m, n, k, workspace_size, floattype };


  
  
  return {"tC0_tA1_tB0_colMaj0_m1601_n64_k2_lda1601_ldb269_ldc269_ws0_f32"};
}

MIOpenGEMM::Offsets get_offsets(){

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
void print_kernel(){
  std::string kernel_string;  
  std::string hyperstring = get_hyperstring();
  auto gg = get_geometry<TFloat>();

  MIOpenGEMM::openclutil::OpenCLDeviceInfo devinfo;
  devinfo.wg_atom_size = 32;
  MIOpenGEMM::hyperparams::Graph graph(gg, devinfo, hyperstring, true); 
  MIOpenGEMM::hyperparams::HyperParams hp(graph);
  bool mowri_verbose = true;
  bool verbose_get_bundle = true;
  std::string mowri_out("");
  MIOpenGEMM::outputwriting::OutputWriter mowri(mowri_verbose, mowri_out != "" , mowri_out);
  auto bundle = MIOpenGEMM::kerngen::get_bundle(hp, gg, mowri, verbose_get_bundle);
  
  
  for (auto & x :  bundle.v_tgks){
    auto dirname = "/home/james/01/" + gg.get_string() + "/" + get_hyperstring() + "/";
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
  


int main(){

  std::string fout("");
  MIOpenGEMM::outputwriting::OutputWriter mowri(true, fout != "" , fout);


  bool test_print = true;
  bool test_benchgemm = false;  
  bool test_find = false;
  bool test_accuracy = true;
  bool test_default = false;

  std::string constraints_string("A_WOS0__B_WOS0");
  
  float allotted_find_time = 1.00;
  unsigned allotted_find_descents = 100;
  unsigned n_runs_per_kernel = 5;
  MIOpenGEMM::SummaryStat sumstat(MIOpenGEMM::Max);
  
  unsigned n_runs_benchgemm = 5;
  
  typedef float tfloat;
  
  
  MIOpenGEMM::Geometry gg = get_geometry<tfloat>();
  MIOpenGEMM::Offsets toff = get_offsets();
  mowri << "generating cpu data ... " << MIOpenGEMM::Flush;
  std::vector<tfloat> v_a;
  std::vector<tfloat> v_b;
  std::vector<tfloat> v_c;
  MIOpenGEMM::setabcw::set_abc<tfloat>(v_a, v_b, v_c, gg, toff);
  mowri << "done." << MIOpenGEMM::Endl;
  const tfloat * c_true_bla = nullptr; 
  if (test_print){
    print_kernel<tfloat>();
  }
  
  if (test_accuracy || test_benchgemm){
    std::string hyperstring = get_hyperstring();
      if (test_accuracy){
        MIOpenGEMM::dev::accuracy_test(hyperstring, gg, toff, v_a.data(), v_b.data(), v_c.data(), c_true_bla, mowri);
    }

    if (test_benchgemm){
      MIOpenGEMM::dev::benchgemm({hyperstring}, n_runs_benchgemm, gg, toff, v_a.data(), v_b.data(), v_c.data(), mowri);
    }
  }
  
  
  MIOpenGEMM::FindParams find_params(allotted_find_time, allotted_find_descents, n_runs_per_kernel, sumstat);
  
  if (test_find){
    auto soln = MIOpenGEMM::dev::find(find_params, v_a.data(), v_b.data(), v_c.data(), constraints_string, gg, toff, mowri);
    std::cout << "\n\n " << soln.get_cache_entry_string() << "\n\n";
  }
  
  if (test_default){
    std::string k_comment("");  
    MIOpenGEMM::openclutil::CommandQueueInContext tgcq(mowri, "in get_default in devtest.cpp");
    auto soln = MIOpenGEMM::get_default(tgcq.command_queue, constraints_string, gg, k_comment, mowri);
    std::cout << soln.hyper_param_string << std::endl;
    MIOpenGEMM::dev::accuracy_test(soln.hyper_param_string, gg, toff, v_a.data(), v_b.data(), v_c.data(), c_true_bla, mowri);
    std::cout << soln.hyper_param_string << std::endl;
  }
  
  return 0;
}


