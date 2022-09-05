//
// Created by test on 2021-06-04.
//

#ifndef LEEFRAMEWORK_DEBUGUTILS_CPP
#define LEEFRAMEWORK_DEBUGUTILS_CPP

#include <iostream>
#include <GL/glew.h>
#include "UtilHeader.h"


GLenum debug_glCheckError(const char* message)
{
    GLenum errorCode;
    while ((errorCode = glGetError()) != GL_NO_ERROR)
    {
        std::string error;
        switch (errorCode)
        {
            case GL_INVALID_ENUM:                  error = "INVALID_ENUM"; break;
            case GL_INVALID_VALUE:                 error = "INVALID_VALUE"; break;
            case GL_INVALID_OPERATION:             error = "INVALID_OPERATION"; break;
            case GL_STACK_OVERFLOW:                error = "STACK_OVERFLOW"; break;
            case GL_STACK_UNDERFLOW:               error = "STACK_UNDERFLOW"; break;
            case GL_OUT_OF_MEMORY:                 error = "OUT_OF_MEMORY"; break;
            case GL_INVALID_FRAMEBUFFER_OPERATION: error = "INVALID_FRAMEBUFFER_OPERATION"; break;
        }
        std::cout << error << " | " << " (" << message << ")" << std::endl;
    }
    return errorCode;
}
#endif

