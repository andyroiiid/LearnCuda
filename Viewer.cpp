// Copyright 2024 Andrew Huang. All Rights Reserved.

#pragma once

#include "Render.cuh"
#include "Shader.h"

#include <GLFW/glfw3.h>
#include <cstdio>
#include <glad/gl.h>

int main()
{
    constexpr int WIDTH = 1920;
    constexpr int HEIGHT = 1080;

    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "LearnCuda", nullptr, nullptr);

    glfwMakeContextCurrent(window);
    gladLoadGL(glfwGetProcAddress);

    GLuint quad = 0;
    glGenBuffers(1, &quad);
    glBindBuffer(GL_ARRAY_BUFFER, quad);

    constexpr float VERTICES[] = {
        -1.0f, -1.0f, 0.0f, 0.0f,
        1.0f, -1.0f, 1.0f, 0.0f,
        -1.0f, 1.0f, 0.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(VERTICES), VERTICES, GL_STATIC_DRAW);

    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(sizeof(float) * 2));

    Shader shader = {
        R"GLSL(
#version 330 core

layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexcoord;

out vec2 vTexcoord;

void main()
{
    gl_Position = vec4(aPos, 0, 1);
    vTexcoord = aTexcoord;
}
)GLSL",
        R"GLSL(
#version 330 core

in vec2 vTexcoord;

uniform sampler2D uTexture;

void main()
{
    gl_FragColor = texture(uTexture, vTexcoord);
}
)GLSL"
    };

    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    float3* pixels = Render(WIDTH, HEIGHT);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGHT, 0, GL_RGB, GL_FLOAT, pixels);
    FreeImage(pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        shader.Use();
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glfwSwapBuffers(window);
    }

    glfwDestroyWindow(window);

    glfwTerminate();

    return 0;
}
