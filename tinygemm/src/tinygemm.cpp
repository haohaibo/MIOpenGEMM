#include <thread>
#include <limits>
#include <chrono>
#include <sstream>
#include <vector> 
#include <algorithm>
#include <map>

#include <tinygemm/tinygemmerror.hpp>
#include <tinygemm/tinygemm.hpp>
#include <tinygemm/consistencychecks.hpp>
#include <tinygemm/outputwriter.hpp>
#include <tinygemm/sizingup.hpp>
#include <tinygemm/openclutil.hpp>
#include <tinygemm/tinygemmsolution.hpp>
#include <tinygemm/hyperparams.hpp>
#include <tinygemm/bundle.hpp>
#include <tinygemm/tinygemmkernel.hpp>
#include <tinygemm/derivedparams.hpp>
#include <tinygemm/architests.hpp>
#include <tinygemm/kernelstring.hpp>

namespace tinygemm{

/* TODO TODO TODO : float4 */

class MultiFloatType{

private:
  double v_d;
  float v_f;  
  
public:
  MultiFloatType(double v): v_d(v), v_f(static_cast<float>(v)) {}
  void * operator [] (char floattype) const{
    return floattype == 'd' ? (void *)(& v_d) : (void * )(& v_f);
  }
  
};


/* TODO : this is exactly like that example from Scott Meyers, return type index by map */
template <typename V>
auto safe_at(const std::map<std::string,V> & m, const std::string & k, std::string hash = "") -> decltype(m.at(k)) {
  
  if (m.count(k) == 0){
    throw tinygemm_error("unrecognised key, " + k + "  (" + hash + ")");
  }
  
  else{
    return m.at(k);
  }

}

static const MultiFloatType m_alpha(default_alpha);
static const MultiFloatType m_beta(default_beta);

class TinyGemmGPUMems{
  private:
    cl_mem a_gpu;
    cl_mem b_gpu;
    cl_mem c_gpu;
    cl_mem workspace_gpu;
  
  public:
    TinyGemmGPUMems(cl_mem a_gpu_, cl_mem b_gpu_, cl_mem c_gpu_, cl_mem workspace_gpu_):a_gpu(a_gpu_), b_gpu(b_gpu_), c_gpu(c_gpu_), workspace_gpu(workspace_gpu_) {}

    cl_mem & operator[](char x){
      if (x == 'a'){
        return a_gpu;
      }
      else if (x == 'b'){
        return b_gpu;
      }
      else if (x == 'c'){
        return c_gpu;
      }
      else if (x == 'w'){
        return workspace_gpu;
      }
      
      else{
        throw tinygemm_error(std::string("unrecognised char passed to operator[] of TinyGemmGPUMems. Should be one of a,b,c,w, not ") + x);
      }
    }
};  

class OpenCLGemmEncapsulator{

public:   
  cl_command_queue command_queue;
  std::string outputfilename;
  const tinygemm::TinyGemmGeometry gg;
  const tinygemm::TinyGemmOffsets toff;
  
  TinyGemmGPUMems gpum;
  
  /* TODO : this belongs somewhere else */
  const std::vector<std::string> possible_basic_types = {"wsa", "wsb", "betac", "main"};
  

private:
  outputwriting::OutputWriter mowri;
  /* vector of times over a set of runs on core loop */
  std::vector<float> v_t_total;

  std::map<std::string, TinyGemmKernel > tk_kernels_map;
  std::vector <TinyGemmKernel *> tk_kernels_active;

  std::vector<std::vector <unsigned > > v_wait_indices;



public:
  OpenCLGemmEncapsulator(
  cl_command_queue command_queue_, 
  const tinygemm::TinyGemmGeometry gg_,
  const tinygemm::TinyGemmOffsets toff_,
  cl_mem a_gpu_,
  cl_mem b_gpu_, 
  cl_mem c_gpu_,
  cl_mem workspace_gpu_,
  std::string outputfilename_,
  bool verbose_):
  
  command_queue(command_queue_), 
  outputfilename(outputfilename_),
  gg(gg_),
  toff(toff_),
  gpum(a_gpu_, b_gpu_, c_gpu_, workspace_gpu_),
  mowri(verbose_, outputfilename.compare("") != 0, outputfilename_)
  {
    

    
    for (auto & x : possible_basic_types){
      tk_kernels_map[x] = TinyGemmKernel(command_queue, x);
    }
    
    run_checks();
  }
  
private:
  void address_check_valid(){
    if (gpum['c'] == gpum['a'] || gpum['c'] == gpum['b']){
      throw tinygemm_error("c should be distinct from a and b for gemm, otherwise race condition arises (one thread writes its result to c before another one has finished reading from c)");
    }
    
    if (gpum['c'] == nullptr){
      throw tinygemm_error("c should not be nullptr");
    }
    
    if (gpum['w'] == nullptr && gg.workspace_size != 0){
      throw tinygemm_error("pointer to workspace memory is the nullptr, but workspace_size is not zero");
    }
    
    if (gpum['w'] != nullptr && gg.workspace_size == 0){
      throw tinygemm_error("pointer to workspace memory is not the nullptr, but workspace_size is zero. if workspace_size is zero please set workspace_gpu to the nullptr to make super clear that there will be no workspace used ");      
    }
    
    if (gpum['w'] != nullptr && (gpum['w'] == gpum['a'] || gpum['w'] == gpum['b'] || gpum['w'] == gpum['c'] ) ){
      throw tinygemm_error("pointer to workspace memory is not the nullptr, and it is the same as one of the a,b,c pointers ");
    }
  }
  
  void address_check_valid_and_reliable(){
    address_check_valid();
    if (gpum['a'] == gpum['b']){
      throw tinygemm_error( "a and b are the same. this will effect kernel run time, not sure if this should be allowed so throwing"); 
    }
  }


  void run_checks(){    
    sizingup::check_sizes_ok_for_unsigned(gg, toff);
  }  

    
  void set_kern_args(const KernelType & type){

    /* parameter order rule: {a, oa, b, ob, c, oc, ws, ows}, alpha, beta */
    std::vector<std::pair<size_t, const void *> > arg_sizes_values;
    
  
    for (auto & x : {'a', 'b', 'c', 'w'}){
      if (type.uses(x) == true){
        arg_sizes_values.emplace_back(sizeof(cl_mem), (void *)&(gpum[x]));
        arg_sizes_values.emplace_back(sizeof(unsigned), &(toff[x]));
      }
    }
    
    if (type.uses_alpha){
      arg_sizes_values.emplace_back(gg.derived.float_size_bytes, m_alpha[gg.floattype]);
    }
    
    if (type.uses_beta){
      arg_sizes_values.emplace_back(gg.derived.float_size_bytes, m_beta[gg.floattype]);      
    }
    
    tk_kernels_map.at(type.basic).set_kernel_args(arg_sizes_values);
    
  }



  
    
    

  bool refresh_needed(const std::string & type, const hyperparams::HyperParams & new_hp, const derivedparams::DerivedParams & new_dp){
    
    //TODO here : check hyper parameters to see if needed a new 
            
    if (type.compare("betac") == 0){
       if (tk_kernels_map.at("betac").is_set() == false && new_dp.main_does_beta_c_inc == 0){
         return true;
       }
       else{
         return false;
       }
    }
    
    else if (type.compare("main") == 0){
      return true;
    }
    
    else if (type.compare("wsa") == 0){
      if (tk_kernels_map.at("wsa").is_set() == false && new_hp.aps.workspace_type != 0){
         return true;
       }
       else{
         return false;
       }
    }

    else if (type.compare("wsb") == 0){
      if (tk_kernels_map.at("wsb").is_set() == false && new_hp.bps.workspace_type != 0){
         return true;
       }
       else{
         return false;
       }
    }

    
    
    else{
      throw tinygemm_error("what is the type of this kernel? Don't recognise it : " + type);
    }
    
  }
  
  void refresh_kernel(const KernelString & ks, const hyperparams::HyperParams & hp, const derivedparams::DerivedParams & dp){

    auto type = ks.type;
    if (refresh_needed(type.basic, hp, dp) == true){
      tk_kernels_map.at(type.basic).update(ks, mowri);
      set_kern_args(type);
    }
  }

    
 
  //TODO : move kernstr in (rvalue ref)
  void setup_tinykernels(const hyperparams::HyperParams & hp, const kerngen::Bundle & bundle ){
    
    
    
    v_wait_indices = bundle.v_wait_indices;
    

    tk_kernels_active.resize(0);
    
    
    
    for (unsigned ksi = 0; ksi < bundle.v_tgks.size(); ++ksi){
      std::string basic = bundle.v_tgks[ksi].type.basic;
      refresh_kernel(bundle.v_tgks[ksi], hp, bundle.dp);
      tk_kernels_active.push_back(&tk_kernels_map[basic]);
    }
  }
  


  void update_run_times(cl_int status){
    
    if (status == CL_SUCCESS){
      
      for (auto & ptr_tk_kernel : tk_kernels_active){
        ptr_tk_kernel->update_times();
      }
      /* end time of last kernel - start time of first kernel */
      v_t_total.push_back(1e-6*(tk_kernels_active.back()->t_end - tk_kernels_active[0]->t_start));
    }
    
    else{
      v_t_total.push_back(std::numeric_limits<float>::max());
    }
  }
  
  void print_run_times(cl_int status){
    
    if (status == CL_SUCCESS){
      mowri << "total time : " <<  v_t_total.back() << "\t (";
      for (unsigned k_ind = 0; k_ind < tk_kernels_active.size(); ++k_ind){
        mowri << " k" << k_ind << ": " << tk_kernels_active[k_ind]->v_times.back() << " ";
      }
      mowri << ") " << "\tGflops/s : " << 2.0 * gg.m * gg.n * gg.k / (v_t_total.back() * 1e6) << Endl;
    }

    else{
      mowri << "elapsed time : " <<  " (max float) \n";
    }
  }


  void reset_v_times(){
    v_t_total.resize(0);
    
    for (auto & ptr_tk_kernel : tk_kernels_active){
      ptr_tk_kernel->reset_times();
    }
  }
  
  
  void core_gemm_loop(size_t n_runs){
    
    reset_v_times();
    
    for (size_t kqq = 0; kqq < n_runs; ++kqq){

      /* This pause should have zero effect, but mysteriously it smooths out the run times between runs when working with certain gpu drivers
       * something to do with overheating  */        
      if (n_runs > 1){
        std::this_thread::sleep_for(std::chrono::milliseconds(0));
      }
  
      /* ***************************************************************************************
       *  Note on timing : the same results have been obtained whth timespec and std::chrono   *
       *  **************************************************************************************/
      
      int status = 10111; 

      for (unsigned k_ind = 0; k_ind < tk_kernels_active.size(); ++k_ind){
      /* At this point, the kernel has been succesfully compiled, 
       * but it is still possible that the resources necessary (LDS etc) are
       * not sufficient on this machine. We catch this case here. 
       * TODO : architects can go some way to catching these before compilation */        
        

        std::vector<cl_event> clevent_waits;

        for (auto & evi : v_wait_indices[k_ind]){
          /* copying cl_events is dangerous. 
           * I have seen that copying them before passed to enqueue 
           * (last parameter) causes problems,
           * this is my idea of what is going on, to confirm: 
           * from cl.h, we see that
           * typedef struct _cl_event *          cl_event,
           * that is cl_event is a pointer to a _cl_event. 
           * when a cl_event address is passed to enqueue,
           * the value if it changes. that is it points to a different _cl_event.
           * thus ev_a = ev_b, enqueue(..., ev_b) 
           * leaves ev_a pointing to a bad place 
           * checking the event is safe:
           * clGetEventInfo takes cl_events by value. 
           * So the moral of the story is : 
           * don't copy cl_events before passing their address  
           * as non-const pointers somewhere!
           * paruse cl.h, sometimes *cl_event is passed as const, sometimes not
           *  */
          clevent_waits.emplace_back(tk_kernels_active[evi]->clevent);
        }
        
        size_t num_events_int_wait_list = clevent_waits.size();
        const cl_event * event_wait_list = num_events_int_wait_list == 0 ? nullptr : clevent_waits.data();
        status = tk_kernels_active[k_ind]->enqueue(num_events_int_wait_list, event_wait_list);

        ///* the in series solution */  
        //if (k_ind == 0){ 
          //status = tk_kernels_active[k_ind]->enqueue();
        //}
        
        //else{
          //status = tk_kernels_active[k_ind]->enqueue(1, &(tk_kernels_active[k_ind - 1]->clevent));
        //}


        if (status == CL_OUT_OF_RESOURCES){
          /* Set the run time(s) and append to vectors */
          mowri << "kernel could not be enqueued, status returned from clEnqueueNDRangeKernel was CL_OUT_OF_RESOURCES (" <<tk_kernels_active[k_ind]->hash << ")" << Endl;
          break;
        }
      }
      
      /* I'm not really sure when to use cl Flush TODO : find out. */
      openclutil::cl_flush(command_queue, "cl flushing in core gemm loop");      
      
      
      
      if (status == CL_SUCCESS){
        /* Wait for kernels to complete */
        openclutil::cl_wait_for_events(1, &(tk_kernels_active.back()->clevent), "with status == CL_SUCCESS in core gemm loops");
      }

      else if (status == CL_OUT_OF_RESOURCES){
        //
      }
      
      else{
        throw std::logic_error("How can this not be CL_SUCCESS or CL_OUT_OF_RESOURCES? Algo prob, come fix. Is status 10111? Maybe there are no kernels?");
      }
      
      update_run_times(status);
      print_run_times(status);
    }
  }

  void deriveability_test(const hyperparams::HyperParams & hp, const std::string & hash){
    auto deriveability = derivedparams::get_deriveability(hp, gg);      
    if (std::get<0>(deriveability) == false){
      throw tinygemm_error(hash + ": the hyper parameters in benchgemm are not consistent, specifically, from get_deriveability \n" + std::get<1>(deriveability));
    }
  }

public:
  void benchgemm(const std::vector<hyperparams::HyperParams> & hps, unsigned n_runs){

    address_check_valid();

    if (n_runs == 0){
      throw tinygemm_error("n_runs to benchgemm should be a positive integer");
    }

    for ( unsigned i = 0; i < hps.size(); ++i) {

      
      mowri << "\nSource kernel " << "(" << i + 1 << "/" << hps.size() << ") "  << hps[i].get_string() << Endl;      
      

      deriveability_test(hps[i], "in benchgemm");

      auto bundle = tinygemm::kerngen::get_bundle(hps[i],gg); 

      auto atr = architests::architecture_specific_tests(command_queue, hps[i], bundle.dp);
      if (std::get<0>(atr) == false){
        throw tinygemm_error(std::get<1>(atr));
      }

      setup_tinykernels(hps[i], bundle); 
      
      mowri << "(benchgemm) geometry  \t:" << gg.get_string()  << "\nEntering the core gemm loops" << Endl;
      core_gemm_loop(n_runs);
    }
  }
  
  
  tinygemm::TinyGemmSolution
  get_default(const bool enforce_deterministic){
    
    hyperparams::HyperParams hp = hyperparams::get_default(gg, enforce_deterministic);
    
    
    deriveability_test(hp, "in get_default");
    
    auto bundle = tinygemm::kerngen::get_bundle(hp,gg); 
    tinygemm::TinyGemmSolutionStatistics tgss(std::numeric_limits<float>::max(), 0, 0);    
    
    return { hp, gg, bundle.dp, tgss, bundle.v_tgks };
    
  }
  
  
  tinygemm::TinyGemmSolution find(float allotted_time, bool enforce_deterministic, unsigned n_runs_per_kernel){
    
    

    
    if (gg.m < 8 || gg.n < 8){
      mowri << "really skinny/thin matrix, returning a default kernel (to be improved) " << Endl;
      return get_default(enforce_deterministic);
    }
      
    
    if (allotted_time <= 0){
      mowri << "Allotted time insufficient for benchmarking, returning default TinyGemmSolution" << Endl;
      return get_default(enforce_deterministic);      
    }

    address_check_valid_and_reliable();
        
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> fp_ms = end - start;
    float elapsed_seconds = fp_ms.count();

    /* we will count how many kernels are successfully generated AND compiled AND benchmarked */
    unsigned global_counter = 0;
        
    /* we track the best TinyGemmSolution found during the search  */    
    std::vector<tinygemm::TinyGemmSolution> path_of_best_solns;
    
    /* In here, we will store all previously considered HyperParams, used to check and ensure that we do not consider a HyperParam more than once */
    std::vector<hyperparams::HyperParams> hyper_front_history;
        
    /* while generating, compiling and benchmarking kernels, we will keep track of the fastest found thus far */
    float best_time = std::numeric_limits<float>::max();
    
    hyperparams::HyperParams best_hp = hyperparams::get_default(gg, enforce_deterministic);
    
    
    



    /* we initialise the `hyper-front' with a single HyperParams, selected based on problem dimensions  */
    std::vector<hyperparams::HyperParams> hyper_front = { hyperparams::get_default(gg, enforce_deterministic) };
    
        
    auto hyper_param_start = hyper_front[0];
 
    bool improvement_found_on_front = true;
    
    /* a hyper front consists of all kernels within a certain "distance of the current best. We start with a front
     * distance of 1, and when this gets into a local mimimum we switch to front distance 2. Front distance 2 kernels
     * are defined is terms of front distance 1 kernels : front distance 2 kernels are just the concatenatition of 
     * the distance 1 kernels from all distance 1 kernels */
    unsigned front_search_horizon = 1;
    
    mowri << "allotted time : " << allotted_time << Endl;
    while (improvement_found_on_front == true){
      
      
      improvement_found_on_front = false;
      mowri << "\nnew hyper front size : " << hyper_front.size() << Endl;

      unsigned hfi = 0;
      while (hfi < hyper_front.size() && improvement_found_on_front == false && elapsed_seconds < allotted_time){
        
        hyperparams::HyperParams hp = hyper_front[hfi];
        
        
        
        /* certain kernels will not be generated, for diverse reasons */
        /* reason 0 : it's already been considered */
        if (std::find(hyper_front_history.begin(), hyper_front_history.end(), hp) != hyper_front_history.end()){
          /* this kernel has already been considered */
        }
        
        else{
          
          hyper_front_history.push_back(hp);
        
          /* reason 1 : the macro tile is too tall */
          if (gg.m < hp.aps.macro_tile_length){
            mowri << "m < aps.macro_tile_length, not considering this kernel" << Endl;
          }
          
          /* reason 2 : the macro tile is too wide */
          else if (gg.n < hp.bps.macro_tile_length){
            mowri << "m < bps.macro_tile_length, not considering this kernel" << Endl;
          }
          
          /* reason 3 : the user requests a deterministic kernel, which cannot be guaranteed */
          else if (enforce_deterministic == true && hp.n_work_items_per_c_elm != 1){
            mowri << "not considering kernels which may be non-deterministic" << Endl;
          }
          /* ************************************************************************ */
  
          /* We will now attempt to generate the kernel */
          else {

            /* attempt to generate the kernel. Certain `bad' kernels are only caught at this stage (set_status.is_good() == false)
             * with tests for hyper-parameter compatibilty in the python script which I don't want to recode here. The main compatibility 
             * issue caught here is that load sizes from global are multiples of the number of work items.  */
            
            auto deriveability = derivedparams::get_deriveability(hp, gg);
            
            if (std::get<0>(deriveability) == true){
              
              auto bundle = tinygemm::kerngen::get_bundle(hp,gg);  
                            
              /* the kernel was succesfully generated, we now compile and benchmark it */
              
              ++global_counter;
              mowri << "\nglobal gen-com-bench : " << global_counter  <<  ".\n" << hp.get_string() << Endl;
              
              setup_tinykernels(hp, bundle);  
              mowri << "(find) geometry : " << gg.get_string()  << "\nEntering the core gemm loops" << Endl; 
              core_gemm_loop(n_runs_per_kernel);
  
              std::sort(v_t_total.begin(), v_t_total.end());
  
              /* Taking the fastest or median? */ 
              float median_time = v_t_total[v_t_total.size()/2]; 
              
              if (std::abs(v_t_total.back() - median_time) / median_time > 0.2) {
                mowri << "tinygemm_warning: large variance in times. " <<  Endl;
              }
              
              end = std::chrono::high_resolution_clock::now();
              fp_ms = end - start;
              elapsed_seconds = fp_ms.count();
                              
              mowri << "median time  : " << median_time << "\t m-Gflops/s : " << 2.0 * gg.m * gg.n * gg.k / (median_time * 1e6) << Endl;
              mowri << "elapsed seconds : " << elapsed_seconds << Endl;
      
              /* A new best kernel !!! we're only interested in an improvement if it's 0.5% or more */
              if (median_time < 0.995*best_time){
                
                improvement_found_on_front = true;
                best_time = median_time;
                best_hp = hp;
                mowri << "---------- NEW BEST TIME FOUND --------- : " << best_time << Endl << "breaking from current hyper front, creating new hyper front " << Endl;
  
                end = std::chrono::high_resolution_clock::now();
                fp_ms = end - start;
                elapsed_seconds = fp_ms.count();
  
                float median_benchmark_gflops = (2. * gg.m * gg.n * gg.k) / (median_time * 10e5);                
                
                tinygemm::TinyGemmSolutionStatistics tgss(median_time, median_benchmark_gflops, elapsed_seconds);
                     
                path_of_best_solns.emplace_back( hp, gg, bundle.dp, tgss, bundle.v_tgks  );
                
                

              }
           
            }
          
            else{
              mowri << "\nSkipping " << hp.get_string() << ", hyper-parameters incompatible. " << Endl;
              mowri << "Specifically, the message from the kernel string setting function was \n`````\n" << std::get<1>(deriveability) << "'''''\n";
            }
          }
        }
        
        ++hfi;
        
        end = std::chrono::high_resolution_clock::now();
        fp_ms = end - start;
        elapsed_seconds = fp_ms.count();
        
      }
      
  
      /* TODO: maybe. add another level of loop here. get_one_aways, then get_two_aways, etc. 
       * what we will have here is that `one' is just rough tile shape, important stuff.*/
      if (improvement_found_on_front == true && front_search_horizon == 1){        
        /* getting all `one-away's */
        hyper_front = best_hp.get_one_aways(gg);
      }

      
      if (improvement_found_on_front == false && front_search_horizon == 1 && elapsed_seconds < allotted_time){
        ++front_search_horizon;

        /* TODO : if you WANT to go onto front 2, uncomment the following. This should be finalised TODO TODO TODO  */        
        const bool jump_to_front_horizon_size_2 = true;
        if (jump_to_front_horizon_size_2 == true){
          improvement_found_on_front = true;
          mowri << "\nSWITCHING TO FRONT HORIZON SIZE 2\n" << Endl;
        }
      }
      
      if (improvement_found_on_front == true && front_search_horizon == 2){        
        /* getting all `two-aways' */
        hyper_front = best_hp.get_two_aways(gg);
      }
      
      if (improvement_found_on_front == false && front_search_horizon == 2){
        /* this is going to cause the end of the search */
      }
      
      if (front_search_horizon != 1 && front_search_horizon != 2){
        throw std::logic_error("front_search_horizon is neither 1 nor 2. This is currently not possible, Broken algorithm, come fix.");        
      }
    }
    
    if (allotted_time <= elapsed_seconds){
      mowri << "stopping the search because the allotted time has been surpassed" << Endl;
    }
    
    else{
      mowri << "stopping the search because a locally minimal kernel has been found" << Endl;
    }
    
    if (path_of_best_solns.size() == 0){
      throw tinygemm_error("\nUser should never see this error, this is an internal problem. Possibly, there were no solutions found. Which is strange, as at least the initial kernel (the initial hyper front) should have been a solution. Either, the initial kernel was not valid (which should not happen unless my filters are broken) or for whatever reason the kernel was not generated or did not compile. Maybe there is some preceding warning printed which sheds light on this? Another possibility is that there was an error in the kernel string generation which I did not think of. ");
    }
  
    mowri << "\nstart kernel : " << hyper_param_start.get_string() << Endl;
    mowri << "best kernel  : " << best_hp.get_string() << Endl;

    mowri << "the kernels along the path the final solution:  " << Endl; 
    mowri << "hyper parameter string                                          \t time when found\t median gflop/s" << Endl;

    for (auto & x : path_of_best_solns){
      mowri <<  x.get_hyper_param_string() << "\t " << x.statistics.solution_discovery_time << "\t\t " << x.statistics.median_benchmark_gflops  << Endl;
    }
    
    mowri <<  path_of_best_solns.back().get_hyper_param_string() << "\t " << elapsed_seconds << "\t\t " << path_of_best_solns.back().statistics.median_benchmark_gflops  << Endl;
    return path_of_best_solns.back();
  }
}; 

openclutil::SafeClMem get_copy(
cl_command_queue command_queue,
cl_mem c,   
const tinygemm::TinyGemmGeometry & gg,
const tinygemm::TinyGemmOffsets & toff,
const std::string & hash
){
  openclutil::SafeClMem c_copied(hash);
  cl_event c_copy_event; 
  size_t n_c = gg.ldc * (gg.tC == gg.isColMajor ? gg.m : gg.n) + toff.oc;
  size_t c_memsize = gg.derived.float_size_bytes*n_c;
  c_copied.clmem = openclutil::cl_create_buffer_from_command_queue(command_queue, CL_MEM_READ_WRITE, c_memsize, NULL, hash + ", in function get_copy of tinygemm");
  openclutil::cl_enqueue_copy_buffer(command_queue, c, c_copied.clmem, 0, 0, c_memsize, 0, NULL, &c_copy_event, hash + ", in function get_copy of tinygemm");
  openclutil::cl_wait_for_events(1, &c_copy_event, "in function find of tinygemm");
  return c_copied;
}







tinygemm::TinyGemmSolution
find(
float allotted_time,
cl_command_queue command_queue,
cl_mem a,   
cl_mem b,
cl_mem c,
cl_mem workspace,
const bool enforce_deterministic,
const tinygemm::TinyGemmGeometry & gg,
const tinygemm::TinyGemmOffsets & toff,
bool verbose, 
std::string logfile, 
bool c_is_const){
  
  
  /* The number of times each kernel is run in find. 
   * consider adding this parameter to user API. */
  unsigned n_runs_per_kernel = 3;

  tinygemm::consistencychecks::check_ldx_mnk_consistent(gg);  

  if (c_is_const == true){
  
    openclutil::SafeClMem c_copied = get_copy(command_queue, c, gg, toff, "copy of c in find");
    OpenCLGemmEncapsulator oger(command_queue, gg, toff, a, b, c_copied.clmem, workspace, logfile, verbose); 
    return oger.find(allotted_time, enforce_deterministic, n_runs_per_kernel);
  }
  
  else{
    OpenCLGemmEncapsulator oger(command_queue, gg, toff, a, b, c, workspace, logfile, verbose); 
    return oger.find(allotted_time, enforce_deterministic, n_runs_per_kernel);
  }
}

tinygemm::TinyGemmSolution
get_default(
const bool enforce_deterministic,
const tinygemm::TinyGemmGeometry & gg,
bool verbose, 
std::string logfile){
  
  OpenCLGemmEncapsulator oger({}, gg, {0,0,0,0,0,0,0}, {}, {}, {}, {}, logfile, verbose); 
  return oger.get_default(enforce_deterministic);
 
}
  
  



void benchgemm(
  cl_command_queue command_queue,
  const std::vector<hyperparams::HyperParams> & hps,
  unsigned n_runs,
  const tinygemm::TinyGemmGeometry & gg,
  const tinygemm::TinyGemmOffsets & toff, 
  cl_mem a_gpu,
  cl_mem b_gpu, 
  cl_mem c_gpu,
  cl_mem workspace_gpu,  
  bool verbose,
  std::string logfile,
  bool c_is_const){
  
  tinygemm::consistencychecks::check_ldx_mnk_consistent(gg);
  if (c_is_const == true){
    openclutil::SafeClMem c_copied = get_copy(command_queue, c_gpu, gg, toff, "copy of c in benchgemm");
    OpenCLGemmEncapsulator oger(command_queue, gg, toff, a_gpu, b_gpu, c_copied.clmem, workspace_gpu, logfile, verbose);
    oger.benchgemm(hps, n_runs);
  }
  
  else{
    OpenCLGemmEncapsulator oger(command_queue, gg, toff, a_gpu, b_gpu, c_gpu, workspace_gpu, logfile, verbose);
    oger.benchgemm(hps, n_runs);
  }
}
  
} //namespace





