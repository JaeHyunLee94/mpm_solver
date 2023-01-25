//
// Created by test on 2022-06-30.
//

#include "Entity.h"
#include <tiny_obj_loader.h>
#include <Partio.h>
#include <fmt/core.h>
void mpm::Entity::loadFromBgeo(const char *filename) {
  // open file
  Partio::ParticlesDataMutable *data = Partio::read(filename);
  if (!data) {
    std::cerr << "Read failed. no particle data loaded\n";
  }
  std::cout << "Number of particles " << data->numParticles() << std::endl;

  _point_list.resize(data->numParticles());
  for (int i = 0; i < data->numAttributes(); i++) {
    Partio::ParticleAttribute attr;
    data->attributeInfo(i, attr);
    std::cout << "attribute[" << i << "] is " << attr.name << std::endl;
  }
  Partio::ParticleAttribute posAttr;

  if (!data->attributeInfo("position", posAttr)
      || (posAttr.type != Partio::FLOAT && posAttr.type != Partio::VECTOR)
      || posAttr.count != 3) {
    std::cerr << "Failed to get proper position attribute" << std::endl;
  }

  for (int i = 0; i < data->numParticles(); ++i) {
    const float *raw_pos = data->data<float>(posAttr, i);
    _point_list[i] = Vec3f{raw_pos[0], raw_pos[1], raw_pos[2]};

  }

}
void mpm::Entity::loadCube(mpm::Vec3f center, mpm::Scalar len, unsigned int particle_num, bool usePoisson) {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-0.5f * len, 0.5f * len);

  _point_list.resize(particle_num);

  if (usePoisson) {
    //TODO: implement
  } else {

    for (auto &p: _point_list) {
      p = center + Vec3f(dis(gen), dis(gen), dis(gen));

    }

  }
  _isEmpty = false;
}
void mpm::Entity::loadSphere(mpm::Vec3f center, mpm::Scalar radius, unsigned int particle_num, bool usePoisson) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(0, 9999);

  _point_list.resize(particle_num);

  if (usePoisson) {
    //TODO: implement
  } else {
    //TODO: implement
  }
  _isEmpty = false;
}
void mpm::Entity::logEntity() {

  printf("Point Size: %llu\n", _point_list.size());
  for (int i = 0; i < _point_list.size(); ++i) {
    printf("%dPoint: (%f %f %f)\n", i, _point_list[i].x(), _point_list[i].y(), _point_list[i].z());
  }

}
std::vector<mpm::Vec3f> &mpm::Entity::getPositionVector() {
  return _point_list;
}

void mpm::Entity::loadFromObjWithPoissonDiskSampling(const char *filename, Scalar radius, float dx) {

  FILE *file = fopen(filename, "r");
  if (file == NULL) {
    printf("Impossible to open the file !\n");
    exit(1);
    return;
  }
  std::vector<::Vec3ui> face;
  std::vector<::Vec3f> v;

  ::Vec3f
      min_box(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()),
      max_box
      (-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());

  while (1) {

    char lineHeader[128];
    // read the first word of the line
    int res = fscanf(file, "%s", lineHeader);
    if (res == EOF)
      break; // EOF = End Of File. Quit the loop.

    // else : parse lineHeader
    if (strcmp(lineHeader, "v") == 0) {
      Vec3f vertex;
      fscanf(file, "%f %f %f\n", &vertex[0], &vertex[1], &vertex[2]);
      min_box[0] = std::min(min_box[0], vertex[0]);
      min_box[1] = std::min(min_box[1], vertex[1]);
      min_box[2] = std::min(min_box[2], vertex[2]);
      max_box[0] = std::max(max_box[0], vertex[0]);
      max_box[1] = std::max(max_box[1], vertex[1]);
      max_box[2] = std::max(max_box[2], vertex[2]);

      _point_list.push_back(vertex);
      v.emplace_back(::Vec3f(vertex[0], vertex[1], vertex[2]));
    } else if (strcmp(lineHeader, "vt") == 0) {

    } else if (strcmp(lineHeader, "vn") == 0) {

    } else if (strcmp(lineHeader, "f") == 0) {
      std::string vertex1, vertex2, vertex3;
      unsigned int vertexIndex[3], uvIndex[3], normalIndex[3];
      int matches = fscanf(file,
                           "%d/%d/%d %d/%d/%d %d/%d/%d\n",
                           &vertexIndex[0],
                           &uvIndex[0],
                           &normalIndex[0],
                           &vertexIndex[1],
                           &uvIndex[1],
                           &normalIndex[1],
                           &vertexIndex[2],
                           &uvIndex[2],
                           &normalIndex[2]);
      fmt::print("face: {} {} {}\n", vertexIndex[0], vertexIndex[1], vertexIndex[2]);
      _face_list.emplace_back(Vec3i(vertexIndex[0]-1, vertexIndex[1]-1, vertexIndex[2]-1));
      face.emplace_back(::Vec3ui(vertexIndex[0]-1, vertexIndex[1]-1, vertexIndex[2]-1));


    }
  }
  fmt::print("Loaded {} points and {} faces from obj file", _point_list.size(), _face_list.size());

  ::Vec3f unit(1, 1, 1);
  int padding = 2;
  min_box -= padding * dx * unit;
  max_box += padding * dx * unit;
  ::Vec3ui sizes = Vec3ui((max_box - min_box) / dx);

  //make SDF
  Array3f phi_grid;
  make_level_set3(face, v, min_box, dx, sizes[0], sizes[1], sizes[2], phi_grid);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0, 0.99);
  float z_len = max_box[2] - min_box[2];
    float y_len = max_box[1] - min_box[1];
    float x_len = max_box[0] - min_box[0];

    for (int i = 0; i < 1000; ++i) {
        ::Vec3f p = min_box + ::Vec3f(dis(gen)*x_len, dis(gen)*y_len, dis(gen)*z_len);
        if (querySDF(p[0], p[1], p[2], min_box, dx, phi_grid)) {
        _point_list.emplace_back(p[0], p[1], p[2]);
        }
    }

  _hasMesh = true;
}
bool mpm::Entity::querySDF(float x, float y, float z, ::Vec3f origin, float dx, Array3f &phi_grid) {
  if (x < origin[0] || y < origin[1] || z < origin[2])
    return false;
  ::Vec3ui index = Vec3ui((x - origin[0]) / dx, (y - origin[1]) / dx, (z - origin[2]) / dx);
  return phi_grid(index[0], index[1], index[2]) < 0;
}
