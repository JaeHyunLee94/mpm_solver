#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec3 normal;
layout(location =3) in vec3 color;

out vec3 v_normal;
out vec3 v_view;
out vec3 v_color;
out vec2 v_uv;


uniform mat4 modelMat,viewMat,projMat;
uniform vec3 eyepos;


void main(){

    vec3 worldPos=(modelMat*vec4(position,1.0)).xyz; // world pos
    gl_Position=projMat*viewMat*vec4(worldPos,1.0);

    v_normal=normalize(transpose(inverse(mat3(modelMat)))*normal); //world normal
    v_view=normalize(eyepos-worldPos); //view vector
    v_color=color;
    v_uv=uv;



}