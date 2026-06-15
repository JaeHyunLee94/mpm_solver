//
// Created by test on 2022-09-19.
//

#ifndef MPM_SOLVER_SRC_SYSTEM_PROFILER_H_
#define MPM_SOLVER_SRC_SYSTEM_PROFILER_H_
#include <ctime>
#include <map>
#include <string>
#include <vector>
class Profiler {

 public:
  Profiler():_labels(0),_values(0),_count(0){

    _labels.reserve(10);
    _values.reserve(10);
  };
  void start(std::string tag);
  void endAndReport(std::string tag);
  void makeArray();
  int& getCount(){return _count;};
  const char** getLabelsPtr(){return _labels.data();};
  double*  getValuesPtr(){return _values.data();};
  std::pair<std::vector<const char*>, std::vector<clock_t>> getLabelsAndValue();
  std::map<std::string, clock_t>& getContainer(){
    return _container;
  }
  std::vector<const char*> _labels;
  std::vector<double> _values;
 private:

  int _count=0;
  std::map<std::string,clock_t> _container;
};



#endif //MPM_SOLVER_SRC_SYSTEM_PROFILER_H_
