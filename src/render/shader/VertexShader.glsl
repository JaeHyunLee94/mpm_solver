#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 particle_pos;//particle pos
layout(location = 3) in float particle_color_weight;//particle vel


out vec3 v_normal;
out vec3 v_view;
out vec3 v_light;
out float v_color_weight;



uniform mat4 modelMat,viewMat,projMat; //model mat unnecessary
uniform vec3 eyepos;
uniform float particle_scale;
uniform vec3 lightsrc;


void main(){

    vec3 worldPos=vec4(position*particle_scale+particle_pos,1.0).xyz; // world pos
    gl_Position=projMat*viewMat*vec4(worldPos ,1.0);

    v_normal=normalize(transpose(inverse(mat3(modelMat)))*normal); //world normal
    v_view=normalize(eyepos-worldPos); //view vector
    v_light=normalize(worldPos-lightsrc); //light vector
    v_color_weight = particle_color_weight;




}