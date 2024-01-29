// Copyright 2024 Andrew Huang. All Rights Reserved.

#pragma once

#include "Vector.cuh"

struct Ray {
    float3 origin;
    float3 direction;

    [[nodiscard]] __host__ __device__ float3 At(const float t) const
    {
        return origin + direction * t;
    }
};

struct Sphere {
    float3 center;
    float radius;

    __host__ __device__ Sphere(const float3& center, const float radius)
        : center(center)
        , radius(radius)
    {
    }

    [[nodiscard]] __host__ __device__ float Hit(const Ray& ray) const
    {
        const float3 oc = ray.origin - center;
        const float a = LengthSquared(ray.direction);
        const float bHalf = Dot(oc, ray.direction);
        const float c = LengthSquared(oc) - radius * radius;
        const float discriminant = bHalf * bHalf - a * c;
        return discriminant < 0 ? -1.0f : (-bHalf - sqrt(discriminant)) / a;
    }
};

struct Triangle {
    float3 p0;
    float3 p1;
    float3 p2;
    float3 normal;

    __host__ __device__ Triangle(const float3& p0, const float3& p1, const float3& p2)
        : p0(p0)
        , p1(p1)
        , p2(p2)
    {
        normal = Normalize(Cross(p1 - p0, p2 - p1));
    }

    [[nodiscard]] __host__ __device__ float Hit(const Ray& ray) const
    {
        const float normalDotDirection = Dot(normal, ray.direction);

        // Cull back faces and parallel faces
        if (normalDotDirection >= 0.0f) {
            return -1.0f;
        }

        const float t = Dot(normal, p0 - ray.origin) / normalDotDirection;
        if (t < 0.0f) {
            return -1.0f;
        }

        const float3 e0 = p1 - p0;
        const float3 e1 = p2 - p1;
        const float3 e2 = p0 - p2;

        const float3 p = ray.At(t);
        const float3 d0 = Cross(e0, p - p0);
        const float3 d1 = Cross(e1, p - p1);
        const float3 d2 = Cross(e2, p - p2);

        return Dot(normal, d0) < 0.0f || Dot(normal, d1) < 0.0f || Dot(normal, d2) < 0.0f ? -1.0f : t;
    }
};
