//
// Created by test on 2022-02-27.
//
#include <Partio.h>

int main(){


    // open file
    Partio::ParticlesDataMutable* data=Partio::read("../../test/test_data/test.bgeo");
    std::cout<<"Number of particles "<<data->numParticles()<<std::endl;

    for(int i=0;i<data->numAttributes();i++){
        Partio::ParticleAttribute attr;
        data->attributeInfo(i,attr);
        std::cout<<"attribute["<<i<<"] is "<<attr.name<<std::endl;
    }
    return 0;
}