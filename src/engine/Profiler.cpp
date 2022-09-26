//
// Created by test on 2022-09-19.
//

#include "Profiler.h"
void Profiler::start(std::string tag) {
  _container[tag]=clock();

}
void Profiler::endAndReport(std::string tag) {

  clock_t end_time= clock();
  clock_t start_time = _container[tag];
  clock_t delta = end_time-start_time;
  _container[tag] =delta;

}

std::pair<std::vector<const char*>, std::vector<clock_t>> Profiler::getLabelsAndValue() {
  std::vector<const char*> labels;
  std::vector<clock_t> values;
  labels.reserve(_container.size());
    values.reserve(_container.size());

  for (auto const& [key, val] : _container)
  {
    labels.push_back(key.c_str());
    values.push_back(val);

  }
  return std::pair<std::vector<const char*>, std::vector<clock_t>>{labels,values};
}
void Profiler::makeArray() {
  _labels.clear();
  _values.clear();
  _labels.reserve(_container.size());
  _values.reserve(_container.size());
  _count=_container.size();
  unsigned long long sum=0;
  for (auto const& [key, val] : _container)
  {
    _labels.push_back(key.c_str());


    sum+=val;
  }
  for (auto const& [key, val] : _container)
  {

    _values.push_back((double)val/sum);

  }

}
