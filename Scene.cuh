#pragma once

#include "Geometry.cuh"

#include <span>

struct Scene {
    static Scene Create(const std::span<const Sphere>& spheres, const std::span<const Triangle>& triangles);

    static void Free(const Scene& scene);

    struct HitResult {
        float t;
        float3 position;
        float3 normal;
    };

    __device__ HitResult Hit(const Ray& ray) const;

    __device__ float3 Trace(Ray ray, int maxBounces, curandState* randomState) const;

    Sphere* m_spheres = nullptr;
    Triangle* m_triangles = nullptr;
    uint32_t m_numSpheres = 0;
    uint32_t m_numTriangles = 0;
};
