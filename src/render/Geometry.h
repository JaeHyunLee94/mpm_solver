//
// Created by test on 2022-08-26.
//

#ifndef MPM_SOLVER_SRC_RENDER_GEOMETRY_H_
#define MPM_SOLVER_SRC_RENDER_GEOMETRY_H_

#include <glm/glm.hpp>

class Mesh {
 public:
  virtual void bind() = 0;
 protected:

  GLuint _VBO;
  GLuint _EBO;
};

class SphereMesh : public Mesh {
 public:
  SphereMesh(float radius, unsigned int stack_count, unsigned int sector_count)
      : _radius(radius), _stackCount(stack_count), _sectorCount(sector_count) {

    const float PI = acos(-1);

    float x, y, z, xy;                              // vertex position
    float nx, ny, nz, lengthInv = 1.0f / radius;    // normal
    float s, t;                                     // texCoord

    float sectorStep = 2 * PI / _sectorCount;
    float stackStep = PI / _stackCount;
    float sectorAngle, stackAngle;

    for (int i = 0; i <= _stackCount; ++i) {
      stackAngle = PI / 2 - i * stackStep;        // starting from pi/2 to -pi/2
      xy = radius * cosf(stackAngle);             // r * cos(u)
      z = radius * sinf(stackAngle);              // r * sin(u)

      // add (sectorCount+1) vertices per stack
      // the first and last vertices have same position and normal, but different tex coords
      for (int j = 0; j <= _sectorCount; ++j) {
        sectorAngle = j * sectorStep;           // starting from 0 to 2pi

        // vertex position
        x = xy * cosf(sectorAngle);             // r * cos(u) * cos(v)
        y = xy * sinf(sectorAngle);
        _vertex.emplace_back(x, y, z);// r * cos(u) * sin(v)


        // normalized vertex normal
        nx = x * lengthInv;
        ny = y * lengthInv;
        nz = z * lengthInv;

        _normal.emplace_back(nx, ny, nz);

      }
    }

    // indices
    //  k1--k1+1
    //  |  / |
    //  | /  |
    //  k2--k2+1
    unsigned int k1, k2;
    for (int i = 0; i < _stackCount; ++i) {
      k1 = i * (_sectorCount + 1);     // beginning of current stack
      k2 = k1 + _sectorCount + 1;      // beginning of next stack

      for (int j = 0; j < _sectorCount; ++j, ++k1, ++k2) {
        // 2 triangles per sector excluding 1st and last stacks
        if (i != 0) {
          // k1---k2---k1+1
          _indices.emplace_back(k1, k2, k1 + 1);
        }

        if (i != (_stackCount - 1)) {
          // k1+1---k2---k2+1
          _indices.emplace_back(k1 + 1, k2, k2 + 1);
        }

      }
    }


    _vertexCount = _vertex.size();
    _triangleCount = _indices.size();
    glGenBuffers(1, &_VBO);
    glGenBuffers(1, &_EBO);
    glBindBuffer(GL_ARRAY_BUFFER, _VBO);

    glBufferData(GL_ARRAY_BUFFER,
                 _vertex.size() * sizeof(glm::vec3) + _normal.size() * sizeof(glm::vec3),
                 nullptr,
                 GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, _vertex.size() * sizeof(glm::vec3), _vertex.data());
    glBufferSubData(GL_ARRAY_BUFFER,
                    _vertex.size() * sizeof(glm::vec3),
                    _vertex.size() * sizeof(glm::vec3) + _normal.size() * sizeof(glm::vec3),
                    _normal.data());

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, _indices.size() * sizeof(glm::uvec3), _indices.data(), GL_STATIC_DRAW);

  };
  void bind() override {
    glBindBuffer(GL_ARRAY_BUFFER, _VBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _EBO);

  };
  unsigned int getVertexCount() const {
    return _vertexCount;
  };
  unsigned int getTriangleCount() const {
    return _triangleCount;
  };

//~SphereMesh();

 private:

  float _radius;
  unsigned int _vertexCount;
  unsigned int _triangleCount;
  std::vector<glm::vec3> _vertex;
  std::vector<glm::vec3> _normal;
  std::vector<glm::uvec3> _indices;
  unsigned int _stackCount;
  unsigned int _sectorCount;

//  void constructVertex();
//  void constructNormal();
//  void constructIndices();

};

#endif //MPM_SOLVER_SRC_RENDER_GEOMETRY_H_
