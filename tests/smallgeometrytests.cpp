/*******************************************************************************
 * Copyright (C) 2017 Advanced Micro Devices, Inc. All rights reserved. 
 *******************************************************************************/
#include <string>
#include <miopengemm/devmiogemm.hpp>
#include <miopengemm/miogemm.hpp>

template <typename TFloat>
void geometrytest(const MIOpenGEMM::Geometry & gg)

//bool isColMajor, bool tA, bool tB, bool tC, size_t m, size_t n, size_t k)
{

  using namespace MIOpenGEMM;
  Offsets offsets = get_padding_offsets(); 

  //FindParams find_params = get_quick_find_params();  
  FindParams find_params(1, 1.14, 2, 200., SummStat::E::MAX);
  std::string constraints_string = "A_WOS2";
  owrite::Writer mowri(Ver::E::ACCURACY, "");
  dev::Boa boa(gg, offsets, mowri);  
  Solution soln = boa.find(find_params, constraints_string);
  std::cout << '\n' << soln.hypas.get_string() << '\n';
  boa.accuracy_test(soln.hypas);

  // TODO : checks on constraints to check for cleary non-derivealboes !!!  
}



int main()
{
  using namespace MIOpenGEMM;
    
  size_t m     = 45;
  size_t k     = 39;
  size_t testi = 0;
  size_t  workspace_size   = 1000*1000; // TODO : checks on workspace size

  for (bool tC : {false, true})
  {
    for (bool isColMajor : {false, true})
    {
      for (bool tA : {false, true})
      {
        for (bool tB : {false, true})
        {
          for (size_t n : {m - 10, m + 10})
          {
            testi += 1;
            k += 1;
            std::cout << "\n\n\ntest " << testi << "/32\n";            
            Geometry gg = get_padded_geometry<float> (isColMajor, tA, tB, tC, m, n, k, workspace_size); 
            std::cout << "<float>  " << gg.get_string() << '\n';
            geometrytest<float>(gg);//isColMajor, tA, tB, tC, m, n, k);
            
            gg = get_padded_geometry<double> (isColMajor, tA, tB, tC, m, n, k, workspace_size); 
            std::cout << "\n\n<double>  " << gg.get_string() << '\n';
            geometrytest<double>(gg);//isColMajor, tA, tB, tC, m, n, k);
          }
        }
      }
    }
  }
  
  std::cout << "\n\n";
  return 0;
}
