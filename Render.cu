#include "Render.cuh"

#include "Geometry.cuh"

#include <glad/gl.h>

unsigned CalcNumBlocks(const unsigned size, const unsigned blockSize)
{
    const unsigned count = size / blockSize;
    return size % blockSize ? count + 1 : count;
}

__device__ float3 Trace(const Ray& ray)
{
    {
        constexpr Sphere sphere {
            { 0.0f, 0.0f, -1.0f },
            0.5f
        };
        const float t = sphere.Hit(ray);
        if (t > 0.0f) {
            return Normalize(ray.At(t) - sphere.center) * 0.5f + 0.5f;
        }
    }

    {
        const Triangle triangle {
            { -1.0f, -1.0f, 0.0f },
            { 1.0f, -1.0f, 0.0f },
            { -1.0f, -1.0f, -2.0f }
        };
        const float t = triangle.Hit(ray);
        if (t > 0.0f) {
            return triangle.normal * 0.5f + 0.5f;
        }
    }

    {
        const Triangle triangle {
            { -1.0f, -1.0f, -2.0f },
            { 1.0f, -1.0f, 0.0f },
            { 1.0f, -1.0f, -2.0f }
        };
        const float t = triangle.Hit(ray);
        if (t > 0.0f) {
            return triangle.normal * 0.5f + 0.5f;
        }
    }

    return Lerp(
        { 1.0f, 1.0f, 1.0f },
        { 0.5f, 0.7f, 1.0f },
        ray.direction.y * 0.5f + 0.5f);
}

__global__ void Render(const int width, const int height, float4* pixels)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const float aspectRatio = static_cast<float>(width) / static_cast<float>(height);

    const float2 uv = {
        static_cast<float>(x) / static_cast<float>(width),
        static_cast<float>(y) / static_cast<float>(height)
    };

    constexpr float3 origin { 0.0f, 0.0f, 0.0f };

    constexpr float focalLength = 1.0f;

    const float3 direction {
        aspectRatio * (uv.x - 0.5f) * 2.0f,
        -(uv.y - 0.5f) * 2.0f,
        -focalLength
    };

    const Ray ray { origin, Normalize(direction) };

    const float3 pixel = Trace(ray);
    pixels[x + y * width] = { pixel.x, pixel.y, pixel.z, 1.0f };
}

void RenderImage(const int width, const int height)
{
    float4* pixels = nullptr;
    cudaMallocManaged(&pixels, sizeof(float4) * width * height);

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
    Render<<<DIM_GRID, DIM_BLOCK>>>(width, height, pixels);

    cudaDeviceSynchronize();

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, pixels);

    cudaFree(pixels);
}
