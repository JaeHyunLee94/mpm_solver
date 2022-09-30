# Material Point Method Engine

This is the simple CUDA, OpenMP based MPM(Material Point Method) Engine.

This Project is for research purpose and is under developing.

## Build instruction
Make sure that your system has CUDA.
```
mkdir build
cd build
cmake ../
make
```
(or give -G option depending on your preference.)

## Features
- [x] [MLS-MPM](https://yzhu.io/publication/mpmmls2018siggraph/paper.pdf)
- [x] Mesh Importer
- [x] CUDA 
- [x] OpenMP
- [x] OpenGL Renderer
- [x] GUI Widget
- [ ] Mesh Exporter
- [ ] Cross platform
- [ ] Mesh Obstacle
- [ ] 2D Cloth, Hair

## Resulting Scenes
These scenes are in the src/scene folder

<!-- <img src = "https://user-images.githubusercontent.com/46246202/193274158-6b4dcf1b-aef7-4ea6-b5a5-99b39527ac15.gif" width="70%"|height="70%"> -->
<img src="https://user-images.githubusercontent.com/46246202/193274158-6b4dcf1b-aef7-4ea6-b5a5-99b39527ac15.gif" width="35%" height="35%"/>
<img src="https://user-images.githubusercontent.com/46246202/193274175-186af6bd-3afb-42ae-a073-0b05b8167c7d.gif" width="35%" height="35%"/>
<img src="https://user-images.githubusercontent.com/46246202/193274094-fcbeb376-9767-4972-a11a-f7e301218623.gif" width="35%" height="35%"/>


## Dependencies
All the dependencies are in the external folder.
