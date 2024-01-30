#pragma once

#include "Render.cuh"

#include <GLFW/glfw3.h>
#include <format>
#include <glad/gl.h>

namespace {

GLuint CreateShader(const GLenum type, const char* source)
{
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint compileStatus = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compileStatus);
    if (compileStatus != GL_TRUE) {
        GLint infoLogLength = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);
        const auto infoLog = new GLchar[infoLogLength];
        glGetShaderInfoLog(shader, infoLogLength, &infoLogLength, infoLog);
        printf("Failed to load shader: %s\n", infoLog);
        delete[] infoLog;
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

GLuint CreateProgram(const GLuint vertexShader, const GLuint fragmentShader)
{
    const GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint linkStatus = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);
    if (linkStatus != GL_TRUE) {
        GLint infoLogLength = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);
        const auto infoLog = new GLchar[infoLogLength];
        printf("Failed to link program: %s\n", infoLog);
        delete[] infoLog;
        glDeleteProgram(program);
        return 0;
    }

    glDetachShader(program, vertexShader);
    glDetachShader(program, fragmentShader);

    return program;
}

void SetupGL()
{
    GLuint quad = 0;
    glGenBuffers(1, &quad);
    glBindBuffer(GL_ARRAY_BUFFER, quad);

    constexpr float VERTICES[] = {
        -1.0f, -1.0f, 0.0f, 1.0f,
        1.0f, -1.0f, 1.0f, 1.0f,
        -1.0f, 1.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 0.0f
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(VERTICES), VERTICES, GL_STATIC_DRAW);

    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(sizeof(float) * 2));

    const auto VERTEX_SOURCE = R"GLSL(
#version 330 core

layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexcoord;

out vec2 vTexcoord;

void main()
{
    gl_Position = vec4(aPos, 0, 1);
    vTexcoord = aTexcoord;
}
)GLSL";

    const auto FRAGMENT_SOURCE = R"GLSL(
#version 330 core

in vec2 vTexcoord;

uniform sampler2D uTexture;

void main()
{
    gl_FragColor = texture(uTexture, vTexcoord);
}
)GLSL";

    const GLuint vertexShader = CreateShader(GL_VERTEX_SHADER, VERTEX_SOURCE);
    const GLuint fragmentShader = CreateShader(GL_FRAGMENT_SHADER, FRAGMENT_SOURCE);
    const GLuint program = CreateProgram(vertexShader, fragmentShader);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    glUseProgram(program);

    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);

    glEnable(GL_FRAMEBUFFER_SRGB);
}

}

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

    SetupGL();

    Framebuffer framebuffer = Framebuffer::Create(WIDTH / 4, HEIGHT / 4);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_SPACE)) {
            framebuffer.frameCount = 0;
        }

        framebuffer.frameCount++;

        const double renderTime = RenderImage(framebuffer);
        glfwSetWindowTitle(window, std::format("Render time = {:.4f}ms", renderTime * 1000).c_str());

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glfwSwapBuffers(window);
    }

    Framebuffer::Free(framebuffer);

    glfwDestroyWindow(window);

    glfwTerminate();

    return 0;
}
