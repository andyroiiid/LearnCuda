// Copyright 2024 Andrew Huang. All Rights Reserved.

#pragma once

struct float3;

float3* Render(int width, int height);

void FreeImage(float3* pixels);
