// Copyright 2024 Andrew Huang. All Rights Reserved.

#pragma once

#include "Vector.cuh"

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

struct Triangle {
    float3 p0;
    float3 p1;
    float3 p2;
    float3 normal;

    __device__ Triangle(const float3& p0, const float3& p1, const float3& p2)
        : p0(p0)
        , p1(p1)
        , p2(p2)
    {
        normal = Normalize(Cross(p1 - p0, p2 - p1));
    }

    [[nodiscard]] __device__ float Hit(const Ray& ray) const
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