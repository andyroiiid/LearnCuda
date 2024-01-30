#include "Render.cuh"

#include "Geometry.cuh"

#include <curand_kernel.h>
#include <glad/gl.h>
#include <span>

unsigned CalcNumBlocks(const unsigned size, const unsigned blockSize)
{
    const unsigned count = size / blockSize;
    return size % blockSize ? count + 1 : count;
}

struct Scene {
    static Scene Create(const std::span<const Sphere>& spheres, const std::span<const Triangle>& triangles)
    {
        Scene scene {};
        cudaMalloc(&scene.m_spheres, spheres.size_bytes());
        cudaMalloc(&scene.m_triangles, triangles.size_bytes());
        cudaMemcpy(scene.m_spheres, spheres.data(), spheres.size_bytes(), cudaMemcpyHostToDevice);
        cudaMemcpy(scene.m_triangles, triangles.data(), triangles.size_bytes(), cudaMemcpyHostToDevice);
        scene.m_numSpheres = spheres.size();
        scene.m_numTriangles = triangles.size_bytes();
        return scene;
    }

    static void Free(const Scene& scene)
    {
        cudaFree(scene.m_spheres);
        cudaFree(scene.m_triangles);
    }

    __device__ float3 Trace(const Ray& ray) const
    {
        float closest = INFINITY;
        float3 normal { 0.0f, 0.0f, 0.0f };

        for (int i = 0; i < m_numSpheres; i++) {
            const Sphere& sphere = m_spheres[i];

            const float t = sphere.Hit(ray);
            if (t < 0.0f || t >= closest) {
                continue;
            }

            closest = t;
            normal = Normalize(ray.At(t) - sphere.center);
        }

        for (int i = 0; i < m_numTriangles; i++) {
            const Triangle& triangle = m_triangles[i];

            const float t = triangle.Hit(ray);
            if (t < 0.0f || t >= closest) {
                continue;
            }

            closest = t;
            normal = triangle.normal;
        }

        if (closest == INFINITY) {
            return Lerp(
                { 1.0f, 1.0f, 1.0f },
                { 0.5f, 0.7f, 1.0f },
                ray.direction.y * 0.5f + 0.5f);
        }

        return normal * 0.5f + 0.5f;
    }

    Sphere* m_spheres = nullptr;
    Triangle* m_triangles = nullptr;
    uint32_t m_numSpheres = 0;
    uint32_t m_numTriangles = 0;
};

// ReSharper disable once CppPassValueParameterByConstReference
__global__ void Render(const int width, const int height, float4* pixels, curandState* randomStates, const int samples, const Scene scene)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const int threadId = x + y * width;
    curandState* randomState = randomStates + threadId;
    curand_init(0, threadId, 0, randomState);

    const float aspectRatio = static_cast<float>(width) / static_cast<float>(height);

    float3 pixel { 0.0f, 0.0f, 0.0f };
    for (int i = 0; i < samples; i++) {
        const float2 uv = {
            (static_cast<float>(x) + curand_uniform(randomState) - 0.5f) / static_cast<float>(width),
            (static_cast<float>(y) + curand_uniform(randomState) - 0.5f) / static_cast<float>(height)
        };

        constexpr float3 origin { 0.0f, 0.0f, 2.0f };

        constexpr float focalLength = 1.0f;

        const float3 direction {
            aspectRatio * (uv.x - 0.5f) * 2.0f,
            -(uv.y - 0.5f) * 2.0f,
            -focalLength
        };

        const Ray ray { origin, Normalize(direction) };

        pixel = pixel + scene.Trace(ray);
    }
    pixel = pixel / samples;

    pixels[threadId] = { pixel.x, pixel.y, pixel.z, 1.0f };
}

void RenderImage(const int width, const int height)
{
    float4* pixels = nullptr;
    curandState* randomStates = nullptr;
    cudaMallocManaged(&pixels, sizeof(float4) * width * height);
    cudaMallocManaged(&randomStates, sizeof(curandState) * width * height);

    const Sphere spheres[] {
        { { 0.0f, 0.0f, 0.0f }, 1.0f },
    };

    const Triangle triangles[] {
        { { -2.0f, -1.0f, 2.0f },
            { 2.0f, -1.0f, 2.0f },
            { -2.0f, -1.0f, -2.0f } },
        { { -2.0f, -1.0f, -2.0f },
            { 2.0f, -1.0f, 2.0f },
            { 2.0f, -1.0f, -2.0f } },
    };

    const Scene scene = Scene::Create(spheres, triangles);

    const dim3 DIM_BLOCK {
        32,
        32,
        1
    };
    const dim3 DIM_GRID {
        CalcNumBlocks(width, DIM_BLOCK.x),
        CalcNumBlocks(height, DIM_BLOCK.y),
        1
    };
    Render<<<DIM_GRID, DIM_BLOCK>>>(width, height, pixels, randomStates, 64, scene);

    cudaDeviceSynchronize();

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, pixels);

    cudaFree(pixels);
    cudaFree(randomStates);
}
