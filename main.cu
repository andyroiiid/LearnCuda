#include <cstdio>

__device__ float3 Trace(const float2 uv)
{
    return float3 { uv.x, uv.y, 0.0f };
}

__global__ void Render(const int width, const int height, float3* pixels)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const float2 uv = {
        static_cast<float>(x) / static_cast<float>(width),
        static_cast<float>(y) / static_cast<float>(height)
    };
    pixels[x + y * width] = Trace(uv);
}

unsigned CalcNumBlocks(const unsigned size, const unsigned blockSize)
{
    const unsigned count = size / blockSize;
    return size % blockSize ? count + 1 : count;
}

void WriteImage(const char* filename, const int width, const int height, const float3* pixels)
{
    FILE* fout = fopen(filename, "w");

    fprintf(fout, "P3\n");
    fprintf(fout, "%d %d\n", width, height);
    fprintf(fout, "255\n");

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const float3 pixel = pixels[x + y * width];
            fprintf(fout, "%d %d %d\n",
                static_cast<int>(255.0f * pixel.x),
                static_cast<int>(255.0f * pixel.y),
                static_cast<int>(255.0f * pixel.z));
        }
    }

    fclose(fout);
}

int main()
{
    constexpr int WIDTH = 1920;
    constexpr int HEIGHT = 1080;
    constexpr int NUM_BYTES = sizeof(float3) * WIDTH * HEIGHT;

    float3* pixels = nullptr;
    cudaMallocManaged(&pixels, NUM_BYTES);

    const dim3 DIM_BLOCK {
        32,
        32,
        1
    };
    const dim3 DIM_GRID {
        CalcNumBlocks(WIDTH, DIM_BLOCK.x),
        CalcNumBlocks(HEIGHT, DIM_BLOCK.y),
        1
    };
    Render<<<DIM_GRID, DIM_BLOCK>>>(WIDTH, HEIGHT, pixels);

    cudaDeviceSynchronize();

    WriteImage("output.ppm", WIDTH, HEIGHT, pixels);

    cudaFree(pixels);

    return 0;
}
