// Copyright 2024 Andrew Huang. All Rights Reserved.

#pragma once

#include <glad/gl.h>

class Shader {
public:
    Shader(const char* vertexSource, const char* fragmentSource);

    ~Shader();

    void Use();

private:
    GLuint m_program = 0;
};
