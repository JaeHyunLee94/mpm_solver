//
// Created by test on 2022-02-27.
//

#include <ccd/ccd.h>
#include <ccd/quat.h> // for work with quaternions

/** Support function for box */
void support(const void *obj, const ccd_vec3_t *dir, ccd_vec3_t *vec)
{

}

int main(int argc, char *argv[])
{

    ccd_t ccd;
    CCD_INIT(&ccd); // initialize ccd_t struct

    // set up ccd_t struct
    ccd.support1       = support; // support function for first object
    ccd.support2       = support; // support function for second object
    ccd.max_iterations = 100;     // maximal number of iterations

    return 0;


}