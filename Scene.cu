#include "Scene.cuh"

Scene Scene::Create(const std::span<const Sphere>& spheres, const std::span<const Triangle>& triangles)
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

void Scene::Free(const Scene& scene)
{
    cudaFree(scene.m_spheres);
    cudaFree(scene.m_triangles);
}

__device__ Scene::HitResult Scene::Hit(const Ray& ray) const
{
    HitResult hit {
        INFINITY,
        { 0.0f, 0.0f, 0.0f },
        { 0.0f, 0.0f, 0.0f }
    };

    constexpr float CLOSEST_HIT = 0.001f;

    for (int i = 0; i < m_numSpheres; i++) {
        const Sphere& sphere = m_spheres[i];

        const float t = sphere.Hit(ray);
        if (t < CLOSEST_HIT || t >= hit.t) {
            continue;
        }

        hit.t = t;
        hit.position = ray.At(t);
        hit.normal = Normalize(hit.position - sphere.center);
    }

    for (int i = 0; i < m_numTriangles; i++) {
        const Triangle& triangle = m_triangles[i];

        const float t = triangle.Hit(ray);
        if (t < CLOSEST_HIT || t >= hit.t) {
            continue;
        }

        hit.t = t;
        hit.position = ray.At(t);
        hit.normal = triangle.normal;
    }

    return hit;
}

__device__ float3 Scene::Trace(Ray ray, const int maxBounces, curandState* randomState) const
{
    int bounces = 0;
    while (bounces < maxBounces) {
        const HitResult hit = Hit(ray);
        if (hit.t == INFINITY) {
            break;
        }
        ray = { hit.position, hit.normal + RandomOnSphere(randomState) };
        bounces++;
    }

    const float3 skybox = Lerp(
        { 1.0f, 1.0f, 1.0f },
        { 0.5f, 0.7f, 1.0f },
        ray.direction.y * 0.5f + 0.5f);
    return skybox * powf(0.5f, bounces);
}
