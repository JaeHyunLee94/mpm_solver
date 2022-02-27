#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>


using namespace std;



void errorCallback(int errorCode, const char* errorDescription)
{
    fprintf(stderr, "Error: %s\n", errorDescription);
}


void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}




int main()
{

    glfwSetErrorCallback(errorCallback);   //#2


    if (!glfwInit()) {

        cerr << "Error: GLFW 초기화 실패" << endl;
        exit(EXIT_FAILURE);
    }



    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); //#3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    glfwWindowHint(GLFW_SAMPLES, 4);


    GLFWwindow* window = glfwCreateWindow(  //#4
            800, // width
            600, // height
            "OpenGL Example", // title
            NULL, NULL);
    if (!window)
        exit(EXIT_FAILURE);


    //#5
    glfwMakeContextCurrent(window);


    glfwSetKeyCallback(window, keyCallback); //#6




    glewExperimental = GL_TRUE;
    GLenum errorCode = glewInit();  //#7
    if (GLEW_OK != errorCode) {

        cerr << "Error: GLEW 초기화 실패 - " << glewGetErrorString(errorCode) << endl;

        glfwTerminate();
        exit(EXIT_FAILURE);
    }



    //#8
    if (!GLEW_VERSION_3_3) {

        cerr << "OpenGL 3.3 API is not available." << endl;

        glfwTerminate();
        exit(EXIT_FAILURE);
    }


    cout << "OpenGL version: " << glGetString(GL_VERSION) << endl;
    cout << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;
    cout << "Vendor: " << glGetString(GL_VENDOR) << endl;
    cout << "Renderer: " << glGetString(GL_RENDERER) << endl;



    //#9
    glfwSwapInterval(1);



    double lastTime = glfwGetTime(); //#11
    int numOfFrames = 0;
    int count = 0;


    while (!glfwWindowShouldClose(window)) {  //#10


        double currentTime = glfwGetTime();
        numOfFrames++;
        if (currentTime - lastTime >= 1.0) {

            printf("%f ms/frame  %d fps \n", 1000.0 / double(numOfFrames), numOfFrames);
            numOfFrames = 0;
            lastTime = currentTime;
        }


        //#12
        if (count % 2 == 0)
            glClearColor(0, 0, 1, 0);
        else
            glClearColor(1, 0, 0, 0);

        glClear(GL_COLOR_BUFFER_BIT);

        count++;


        glfwSwapBuffers(window); //#13
        glfwPollEvents(); //#14

    }


    glfwTerminate();  //#15

    exit(EXIT_SUCCESS);
}