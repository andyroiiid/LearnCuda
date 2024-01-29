#pragma once

#include <vector_types.h>

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
