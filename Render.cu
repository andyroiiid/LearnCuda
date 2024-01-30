#include "Render.cuh"

#include "Scene.cuh"

#include <GLFW/glfw3.h>
#include <glad/gl.h>

unsigned CalcNumBlocks(const unsigned size, const unsigned blockSize)
{
    const unsigned count = size / blockSize;
    return size % blockSize ? count + 1 : count;
}

__global__ void Render(
    // ReSharper disable once CppPassValueParameterByConstReference
    const Framebuffer framebuffer,
    const int maxBounces,
    // ReSharper disable once CppPassValueParameterByConstReference
    const Scene scene)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= framebuffer.width || y >= framebuffer.height) {
        return;
    }

    const int threadId = x + y * framebuffer.width;
    curandState* randomState = framebuffer.randomStates + threadId;
    curand_init(framebuffer.frameCount, threadId, 0, randomState);

    const float aspectRatio = static_cast<float>(framebuffer.width) / static_cast<float>(framebuffer.height);

    const float2 uv = {
        (static_cast<float>(x) + curand_uniform(randomState) - 0.5f) / static_cast<float>(framebuffer.width),
        (static_cast<float>(y) + curand_uniform(randomState) - 0.5f) / static_cast<float>(framebuffer.height)
    };

    constexpr float3 origin { 0.0f, 0.0f, 2.0f };

    constexpr float focalLength = 1.0f;

    const float3 direction {
        aspectRatio * (uv.x - 0.5f) * 2.0f,
        -(uv.y - 0.5f) * 2.0f,
        -focalLength
    };

    const Ray ray { origin, Normalize(direction) };

    float3& pixel = framebuffer.pixels[threadId];
    const float3 sample = scene.Trace(ray, maxBounces, randomState);
    pixel = Lerp(pixel, sample, 1.0f / static_cast<float>(framebuffer.frameCount + 1));
}

double RenderImage(const Framebuffer& framebuffer)
{
    const Sphere spheres[] {
        { { 0.0f, 0.0f, 0.0f }, 1.0f },
    };

    const Triangle triangles[] {
        { { -20.0f, -1.0f, 20.0f },
            { 20.0f, -1.0f, 20.0f },
            { -20.0f, -1.0f, -20.0f } },
        { { -20.0f, -1.0f, -20.0f },
            { 20.0f, -1.0f, 20.0f },
            { 20.0f, -1.0f, -20.0f } },
    };

    const Scene scene = Scene::Create(spheres, triangles);

    const dim3 DIM_BLOCK {
        16,
        16,
        1
    };
    const dim3 DIM_GRID {
        CalcNumBlocks(framebuffer.width, DIM_BLOCK.x),
        CalcNumBlocks(framebuffer.height, DIM_BLOCK.y),
        1
    };
    const double prevTime = glfwGetTime();
    Render<<<DIM_GRID, DIM_BLOCK>>>(framebuffer, 8, scene);
    cudaDeviceSynchronize();
    const double currTime = glfwGetTime();

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB32F,
        framebuffer.width,
        framebuffer.height,
        0,
        GL_RGB,
        GL_FLOAT,
        framebuffer.pixels);

    return currTime - prevTime;
}
