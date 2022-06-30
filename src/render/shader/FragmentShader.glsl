#version 330 core

// Interpolated values from the vertex shaders
in vec3 v_normal;
in vec3 v_view;
in vec3 v_color;
in vec2 v_uv;

// Ouput data
out vec4 frag_color;

//// parameter for phong shader

//light property
uniform vec3 Kd,Ka,Ks,Ke;
uniform vec3 Sa,Ss,Sd;
uniform vec3 lightdir;
uniform float sh;
uniform bool has_texture;
uniform sampler2D diffuse_map;

//material property




void main(){

    vec3 normal =  normalize(v_normal);
    vec3 view = normalize(v_view);
    vec3 light = normalize(lightdir);

    vec3 diff;
    if(has_texture){
        diff=texture(diffuse_map,v_uv).rgb;
    }
    else{
        diff=max(dot(normal,-light),0.0)*Sd*Kd;
    }
    //


    vec3 refl=2.0*normal*dot(normal,light)-light;
    vec3 spec=pow(max(dot(refl,v_view),0),sh)*Ss*Ks;

    vec3 ambi=Ka*Sa;

    //frag_color=vec4(0.9,0.2,0.1,1.0);

    frag_color=vec4(diff+spec+ambi+Ke,1.0);
    //frag_color=vec4(normal,1.0);



}