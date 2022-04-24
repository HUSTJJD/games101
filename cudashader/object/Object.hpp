//
// Created by LEI XU on 5/13/19.
//
#pragma once
#ifndef CUDASHADER_OBJECT_H
#define CUDASHADER_OBJECT_H

#include "../common/Intersection.hpp"
#include "../common/Ray.hpp"
#include "../common/Vector.hpp"
#include "../common/global.hpp"
#include "../partition/Bounds3.hpp"

class Object
{
public:
    Object() {}
    virtual ~Object() {}
    virtual bool intersect(const Ray &ray) = 0;
    virtual bool intersect(const Ray &ray, float &, uint32_t &) const = 0;
    virtual Intersection getIntersection(Ray _ray) = 0;
    virtual void getSurfaceProperties(const Vector3f &, const Vector3f &, const uint32_t &, const Vector2f &, Vector3f &, Vector2f &) const = 0;
    virtual Vector3f evalDiffuseColor(const Vector2f &) const = 0;
    virtual Bounds3 getBounds() = 0;
    virtual float getArea() = 0;
    virtual void Sample(Intersection &pos, float &pdf) = 0;
    virtual bool hasEmit() = 0;
};

#endif // CUDASHADER_OBJECT_H
