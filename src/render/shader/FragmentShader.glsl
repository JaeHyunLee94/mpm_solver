#version 330 core

// Interpolated values from the vertex shaders
in vec3 v_normal;
in vec3 v_view;
in vec3 v_light;
in float v_color_weight;

// Ouput data
out vec4 frag_color;

//// parameter for phong shader

//light property
uniform vec3 Kd,Ka,Ks,Ke;
uniform vec3 Sa,Ss,Sd;
uniform bool isUseRainBowMap;
uniform float sh;
//material property


vec3 rainBowColorMap(float x){
    vec3 color;
    if(x<0.25){
        color = vec3(0.0f,4*x,1.0f);
    }else if (x<0.5){
        color = vec3(0.0f,1.0f,2.0f-4*x);
    }else if (x<0.75){
        color = vec3(4*x-2.0f, 1.0f, 0.0f);
    }else{
        color = vec3(1.0f, 4.0f-4.0f*x, 0.0f);
    }
    return color;
}


void main(){

    vec3 normal =  normalize(v_normal);
    vec3 view = normalize(v_view);
    vec3 light = normalize(v_light);

    vec3 diff;
    vec3 particle_color;
    if(isUseRainBowMap){
        particle_color = rainBowColorMap(v_color_weight);
    }else{
        particle_color = Kd;
    }

//    diff=max(dot(normal,-light),0.0)*Sd*Kd;
    diff=max(dot(normal,-light),0.0)*Sd*particle_color;

    //


    vec3 refl=2.0*normal*dot(normal,light)-light;
    vec3 spec=pow(max(dot(refl,v_view),0),sh)*Ss*Ks;

    vec3 ambi=Ka*Sa;

   // frag_color=vec4(ambi,1.0);

    frag_color=vec4(diff+spec+ambi+Ke,1.0);
    //frag_color=vec4(normal,1.0);



}