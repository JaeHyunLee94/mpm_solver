#version 330 core

// Interpolated values from the vertex shaders
in vec3 v_normal;
in vec3 v_view;
in vec3 v_light;


// Ouput data
out vec4 frag_color;

//// parameter for phong shader

//light property
uniform vec3 Kd,Ka,Ks,Ke;
uniform vec3 Sa,Ss,Sd;

uniform float sh;
uniform vec3 particle_color;
//material property




void main(){

    vec3 normal =  normalize(v_normal);
    vec3 view = normalize(v_view);
    vec3 light = normalize(v_light);

    vec3 diff;

    diff=max(dot(normal,-light),0.0)*Sd*Kd;

    //


    vec3 refl=2.0*normal*dot(normal,light)-light;
    vec3 spec=pow(max(dot(refl,v_view),0),sh)*Ss*Ks;

    vec3 ambi=Ka*Sa;

   // frag_color=vec4(ambi,1.0);

    frag_color=vec4(diff+spec+ambi+Ke,1.0);
    //frag_color=vec4(normal,1.0);



}