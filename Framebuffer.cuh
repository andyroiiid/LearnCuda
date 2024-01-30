#pragma once

struct float3;
typedef struct curandStateXORWOW curandState;

struct Framebuffer {
    static Framebuffer Create(int width, int height);

    static void Free(const Framebuffer& framebuffer);

    int frameCount = 0;
    int width = 0;
    int height = 0;
    float3* pixels = nullptr;
    curandState* randomStates = nullptr;
};
