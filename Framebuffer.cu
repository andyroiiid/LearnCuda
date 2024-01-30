#include "Framebuffer.cuh"

#include <curand_kernel.h>

Framebuffer Framebuffer::Create(const int width, const int height)
{
    Framebuffer framebuffer {};
    framebuffer.width = width;
    framebuffer.height = height;
    cudaMallocManaged(&framebuffer.pixels, sizeof(float3) * width * height);
    cudaMallocManaged(&framebuffer.randomStates, sizeof(curandState) * width * height);
    return framebuffer;
}

void Framebuffer::Free(const Framebuffer& framebuffer)
{
    cudaFree(framebuffer.pixels);
    cudaFree(framebuffer.randomStates);
}
