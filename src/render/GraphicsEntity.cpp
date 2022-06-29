//
// Created by Lee on 2021/07/11.
//

#include "GraphicsEntity.hpp"
#include "../Physics/PhysicsEntity.hpp"

void GraphicsEntity::bind() {
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_EBO);
}
