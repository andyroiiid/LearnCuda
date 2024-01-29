#include "Render.cuh"

#include <glad/gl.h>

#pragma region Vector Operators

inline __device__ float3 operator-(const float3& a)
{
    return { -a.x, -a.y, -a.z };
}

inline __device__ float3 operator+(const float3& a, const float3& b)
{
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

inline __device__ float3 operator+(const float3& a, const float b)
{
    return { a.x + b, a.y + b, a.z + b };
}

inline __device__ float3 operator+(const float a, const float3& b)
{
    return b + a;
}

inline __device__ float3 operator-(const float3& a, const float3& b)
{
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

inline __device__ float3 operator-(const float3& a, const float b)
{
    return { a.x - b, a.y - b, a.z - b };
}

inline __device__ float3 operator-(const float a, const float3& b)
{
    return b - a;
}

inline __device__ float3 operator*(const float3& a, const float3& b)
{
    return { a.x * b.x, a.y * b.y, a.z * b.z };
}

inline __device__ float3 operator*(const float3& a, const float b)
{
    return { a.x * b, a.y * b, a.z * b };
}

inline __device__ float3 operator*(const float a, const float3& b)
{
    return b * a;
}

inline __device__ float3 operator/(const float3& a, const float3& b)
{
    return { a.x / b.x, a.y / b.y, a.z / b.z };
}

inline __device__ float3 operator/(const float3& a, const float b)
{
    return { a.x / b, a.y / b, a.z / b };
}

inline __device__ float Dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ float3 Cross(const float3& a, const float3& b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

inline __device__ float LengthSquared(const float3& a)
{
    return Dot(a, a);
}

inline __device__ float Length(const float3& a)
{
    return sqrt(LengthSquared(a));
}

inline __device__ float Distance(const float3& a, const float3& b)
{
    return Length(a - b);
}

inline __device__ float3 Normalize(const float3& a)
{
    return a / Length(a);
}

inline __device__ float3 Lerp(const float3& a, const float3& b, const float t)
{
    return a + (b - a) * t;
}

#pragma endregion

unsigned CalcNumBlocks(const unsigned size, const unsigned blockSize)
{
    const unsigned count = size / blockSize;
    return size % blockSize ? count + 1 : count;
}

struct Ray {
    float3 origin;
    float3 direction;

    [[nodiscard]] __device__ float3 At(const float t) const
    {
        return origin + direction * t;
    }

    [[nodiscard]] __device__ float HitSphere(const float3& center, const float radius) const
    {
        const float3 oc = origin - center;
        const float a = LengthSquared(direction);
        const float b = 2.0f * Dot(oc, direction);
        const float c = LengthSquared(oc) - radius * radius;
        const float discriminant = b * b - 4 * a * c;
        return discriminant < 0 ? -1.0f : (-b - sqrt(discriminant)) / (2.0f * a);
    }
};

struct Sphere {
    float3 center;
    float radius;

    [[nodiscard]] __device__ float Hit(const Ray& ray) const
    {
        return ray.HitSphere(center, radius);
    }
};

__device__ float3 Trace(const Ray& ray)
{
    constexpr Sphere sphere { { 0.0f, 0.0f, -1.0f }, 0.5f };
    const float t = sphere.Hit(ray);
    if (t > 0.0f) {
        return Normalize(ray.At(t) - sphere.center) * 0.5 + 0.5f;
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