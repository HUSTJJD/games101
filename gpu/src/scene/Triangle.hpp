#ifndef GPU_TRIANGLE_H
#define GPU_TRIANGLE_H
#include <fstream>

#include "../common/Setting.h"
#include "Material.cuh"

struct Vertex {
  float3 point, normal;
  float2 uv;
  Vertex(const float3 p, const float3 n, const float2 u)
      : point(p), normal(n), uv(u) {}
  Vertex()
      : point(make_float3(0, 0, 0)),
        normal(make_float3(0, 0, 0)),
        uv(make_float2(0, 0)) {}
};

struct Triangle {
  Vertex v1, v2, v3;
  int mat;
  Triangle() {}
  Triangle(Vertex a, Vertex b, Vertex c, int m) : v1(a), v2(b), v3(c), mat(m) {}
};

inline Vertex ReadVertex(std::ifstream* read_stream, float size) {
  Vertex vertex;
  read_stream->read(reinterpret_cast<char*>(&vertex.point.x), sizeof(float));
  read_stream->read(reinterpret_cast<char*>(&vertex.point.y), sizeof(float));
  read_stream->read(reinterpret_cast<char*>(&vertex.point.z), sizeof(float));
  read_stream->read(reinterpret_cast<char*>(&vertex.normal.x), sizeof(float));
  read_stream->read(reinterpret_cast<char*>(&vertex.normal.y), sizeof(float));
  read_stream->read(reinterpret_cast<char*>(&vertex.normal.z), sizeof(float));
  read_stream->read(reinterpret_cast<char*>(&vertex.uv.x), sizeof(float));
  read_stream->read(reinterpret_cast<char*>(&vertex.uv.y), sizeof(float));
  vertex.point *= size;
  return vertex;
}
#endif  // GPU_TRIANGLE_H