#pragma once

#include <curand_kernel.h>
#include <math_constants.h>

inline __host__ __device__ float3 operator-(const float3& a)
{
    return { -a.x, -a.y, -a.z };
}

inline __host__ __device__ float3 operator+(const float3& a, const float3& b)
{
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

inline __host__ __device__ float3 operator+(const float3& a, const float b)
{
    return { a.x + b, a.y + b, a.z + b };
}

inline __host__ __device__ float3 operator+(const float a, const float3& b)
{
    return b + a;
}

inline __host__ __device__ float3 operator-(const float3& a, const float3& b)
{
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

inline __host__ __device__ float3 operator-(const float3& a, const float b)
{
    return { a.x - b, a.y - b, a.z - b };
}

inline __host__ __device__ float3 operator-(const float a, const float3& b)
{
    return b - a;
}

inline __host__ __device__ float3 operator*(const float3& a, const float3& b)
{
    return { a.x * b.x, a.y * b.y, a.z * b.z };
}

inline __host__ __device__ float3 operator*(const float3& a, const float b)
{
    return { a.x * b, a.y * b, a.z * b };
}

inline __host__ __device__ float3 operator*(const float a, const float3& b)
{
    return b * a;
}

inline __host__ __device__ float3 operator/(const float3& a, const float3& b)
{
    return { a.x / b.x, a.y / b.y, a.z / b.z };
}

inline __host__ __device__ float3 operator/(const float3& a, const float b)
{
    return { a.x / b, a.y / b, a.z / b };
}

inline __host__ __device__ float Dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float3 Cross(const float3& a, const float3& b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

inline __host__ __device__ float LengthSquared(const float3& a)
{
    return Dot(a, a);
}

inline __host__ __device__ float Length(const float3& a)
{
    return sqrtf(LengthSquared(a));
}

inline __host__ __device__ float Distance(const float3& a, const float3& b)
{
    return Length(a - b);
}

inline __host__ __device__ float3 Normalize(const float3& a)
{
    return a / Length(a);
}

inline __host__ __device__ float3 Lerp(const float3& a, const float3& b, const float t)
{
    return a + (b - a) * t;
}

inline __device__ float3 RandomOnSphere(curandState* randomState)
{
    float sinLatitude = 0.0f;
    float cosLatitude = 0.0f;
    float sinLongitude = 0.0f;
    float cosLongitude = 0.0f;
    sincosf(acosf(2.0f * curand_uniform(randomState) - 1.0f) - CUDART_PI_F / 2.0f, &sinLatitude, &cosLatitude);
    sincospif(2.0f * curand_uniform(randomState), &sinLongitude, &cosLongitude);
    return {
        cosLatitude * cosLongitude,
        cosLatitude * sinLongitude,
        sinLatitude
    };
}

inline __device__ float3 RandomInSphere(curandState* randomState)
{
    const float radius = cbrtf(curand_uniform(randomState));
    return RandomOnSphere(randomState) * radius;
}

inline __device__ float3 RandomOnHemisphere(curandState* randomState, const float3& direction)
{
    const float3 v = RandomOnSphere(randomState);
    return Dot(v, direction) >= 0.0f ? v : -v;
}

inline __device__ float3 RandomInHemisphere(curandState* randomState, const float3& direction)
{
    const float3 v = RandomInSphere(randomState);
    return Dot(v, direction) >= 0.0f ? v : -v;
}
