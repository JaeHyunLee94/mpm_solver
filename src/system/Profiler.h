//
// Created by test on 2022-09-19.
//

#ifndef MPM_SOLVER_SRC_SYSTEM_PROFILER_H_
#define MPM_SOLVER_SRC_SYSTEM_PROFILER_H_

class Profiler {
  //Singleton class
 private:
  Profiler() {}
  Profiler(const Profiler& ref) {}
  Profiler& operator=(const Profiler& ref) {}
  ~Profiler() {}
 public:
  static Profiler& getIncetance() {
    static Profiler s;
    return s;
  }
};



#endif //MPM_SOLVER_SRC_SYSTEM_PROFILER_H_
