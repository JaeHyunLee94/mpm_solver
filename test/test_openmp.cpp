//
// Created by test on 2022-09-13.
//

#include <iostream>
#include <omp.h>

int main()
{
#pragma omp parallel num_threads(3)
  {
    int id = omp_get_thread_num();
    std::cout << "Greetings from process " << id << std::endl;
  }
  std::cout << "parallel for ends " << std::endl;
  return EXIT_SUCCESS;
}