//
// Created by Göksu Güvendiren on 2019-05-14.
//

#ifndef CUDASHADER_SCENE_H
#define CUDASHADER_SCENE_H
#include "../common/AreaLight.hpp"
#include "../common/Light.hpp"
#include "../common/Ray.hpp"
#include "../common/Vector.hpp"
#include "../material/Material.hpp"
#include "../object/Object.hpp"
#include "../object/Triangle.hpp"
#include "../partition/BVH.hpp"
#include <vector>

class Scene
{
public:
    // setting up options
    int width = 1280;
    int height = 960;
    double fov = 10;
    Vector3f backgroundColor = Vector3f(0.235294, 0.67451, 0.843137);
    int maxDepth = 1;
    float RussianRoulette = 0.8;
    BVHAccel *bvh;

    Scene(int w, int h) : width(w), height(h)
    {
    }
    ~Scene()
    {
        for (auto obj : objects)
        {
            delete obj;
        }
    }
    void Add(Object *object) { objects.push_back(object); }
    void Add(std::unique_ptr<Light> light) { lights.push_back(std::move(light)); }

    const std::vector<Object *> &get_objects() const { return objects; }
    const std::vector<std::unique_ptr<Light>> &get_lights() const { return lights; }
    Intersection intersect(const Ray &ray) const;
    void buildScene(std::string path);
    void buildBVH();
    Vector3f castRay(const Ray &ray, int depth) const;
    void sampleLight(Intersection &pos, float &pdf) const;
    bool trace(const Ray &ray, const std::vector<Object *> &objects, float &tNear, uint32_t &index, Object **hitObject);

    std::tuple<Vector3f, Vector3f> HandleAreaLight(const AreaLight &light, const Vector3f &hitPoint, const Vector3f &N,
                                                   const Vector3f &shadowPointOrig,
                                                   const std::vector<Object *> &objects, uint32_t &index,
                                                   const Vector3f &dir, float specularExponent);

    // creating the scene (adding objects and lights)
    std::vector<Object *> objects;

    std::vector<std::unique_ptr<Light>> lights;

    // Compute reflection direction
    Vector3f reflect(const Vector3f &I, const Vector3f &N) const
    {
        return I - 2 * dotProduct(I, N) * N;
    }

    // Compute refraction direction using Snell's law
    //
    // We need to handle with care the two possible situations:
    //
    //    - When the ray is inside the object
    //
    //    - When the ray is outside.
    //
    // If the ray is outside, you need to make cosi positive cosi = -N.I
    //
    // If the ray is inside, you need to invert the refractive indices and negate the normal N
    Vector3f refract(const Vector3f &I, const Vector3f &N, const float &ior) const
    {
        float cosi = clamp(-1, 1, dotProduct(I, N));
        float etai = 1, etat = ior;
        Vector3f n = N;
        if (cosi < 0)
        {
            cosi = -cosi;
        }
        else
        {
            std::swap(etai, etat);
            n = -N;
        }
        float eta = etai / etat;
        float k = 1 - eta * eta * (1 - cosi * cosi);
        return k < 0 ? 0 : eta * I + (eta * cosi - sqrtf(k)) * n;
    }

    // Compute Fresnel equation
    //
    // \param I is the incident view direction
    //
    // \param N is the normal at the intersection point
    //
    // \param ior is the material refractive index
    //
    // \param[out] kr is the amount of light reflected
    void fresnel(const Vector3f &I, const Vector3f &N, const float &ior, float &kr) const
    {
        float cosi = clamp(-1, 1, dotProduct(I, N));
        float etai = 1, etat = ior;
        if (cosi > 0)
        {
            std::swap(etai, etat);
        }
        // Compute sini using Snell's law
        float sint = etai / etat * sqrtf(std::max(0.f, 1 - cosi * cosi));
        // Total internal reflection
        if (sint >= 1)
        {
            kr = 1;
        }
        else
        {
            float cost = sqrtf(std::max(0.f, 1 - sint * sint));
            cosi = fabsf(cosi);
            float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
            float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
            kr = (Rs * Rs + Rp * Rp) / 2;
        }
        // As a consequence of the conservation of energy, transmittance is given by:
        // kt = 1 - kr;
    }
};

void Scene::buildBVH()
{
    printf(" - Generating BVH...\n\n");
    this->bvh = new BVHAccel(objects, 1, BVHAccel::SplitMethod::NAIVE);
}

Intersection Scene::intersect(const Ray &ray) const
{
    return this->bvh->Intersect(ray);
}

void Scene::sampleLight(Intersection &pos, float &pdf) const
{
    float emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k)
    {
        if (objects[k]->hasEmit())
        {
            emit_area_sum += objects[k]->getArea();
        }
    }
    float p = get_random_float() * emit_area_sum;
    emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k)
    {
        if (objects[k]->hasEmit())
        {
            emit_area_sum += objects[k]->getArea();
            if (p <= emit_area_sum)
            {
                objects[k]->Sample(pos, pdf);
                break;
            }
        }
    }
}

bool Scene::trace(
    const Ray &ray,
    const std::vector<Object *> &objects,
    float &tNear, uint32_t &index, Object **hitObject)
{
    *hitObject = nullptr;
    for (uint32_t k = 0; k < objects.size(); ++k)
    {
        float tNearK = kInfinity;
        uint32_t indexK;
        Vector2f uvK;
        if (objects[k]->intersect(ray, tNearK, indexK) && tNearK < tNear)
        {
            *hitObject = objects[k];
            tNear = tNearK;
            index = indexK;
        }
    }

    return (*hitObject != nullptr);
}

// Implementation of Path Tracing
Vector3f Scene::castRay(const Ray &ray, int depth) const
{
    // TO DO Implement Path Tracing Algorithm here
    Intersection inter = intersect(ray);

    if (inter.happened)
    {
        // 如果射线第一次打到光源，直接返回
        if (inter.m->hasEmission())
        {
            if (depth == 0)
            {
                return inter.m->getEmission();
            }
            else
                return Vector3f(0, 0, 0);
        }

        Vector3f L_dir(0, 0, 0);
        Vector3f L_indir(0, 0, 0);

        // 随机 sample 灯光，用该 sample 的结果判断射线是否击中光源
        Intersection lightInter;
        float pdf_light = 0.0f;
        sampleLight(lightInter, pdf_light);

        // 物体表面法线
        auto &N = inter.normal;
        // 灯光表面法线
        auto &NN = lightInter.normal;

        auto &objPos = inter.coords;
        auto &lightPos = lightInter.coords;

        auto diff = lightPos - objPos;
        auto lightDir = diff.normalized();
        float lightDistance = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;

        Ray light(objPos, lightDir);
        Intersection light2obj = intersect(light);

        // 如果反射击中光源
        if (light2obj.happened && (light2obj.coords - lightPos).norm() < 1e-2)
        {
            Vector3f f_r = inter.m->eval(ray.direction, lightDir, N);
            L_dir = lightInter.emit * f_r * dotProduct(lightDir, N) * dotProduct(-lightDir, NN) / lightDistance / pdf_light;
        }

        if (get_random_float() < RussianRoulette)
        {
            Vector3f nextDir = inter.m->sample(ray.direction, N).normalized();

            Ray nextRay(objPos, nextDir);
            Intersection nextInter = intersect(nextRay);
            if (nextInter.happened && !nextInter.m->hasEmission())
            {
                float pdf = inter.m->pdf(ray.direction, nextDir, N);
                Vector3f f_r = inter.m->eval(ray.direction, nextDir, N);
                L_indir = castRay(nextRay, depth + 1) * f_r * dotProduct(nextDir, N) / pdf / RussianRoulette;
            }
        }

        return L_dir + L_indir;
    }

    return Vector3f(0, 0, 0);
}

void Scene::buildScene(std::string path)
{
    std::cout << "Building scene " << std::endl;
    Material *red = new Material(Vector3f(0.63f, 0.065f, 0.05f));

    Material *green = new Material(Vector3f(0.14f, 0.45f, 0.091f));

    Material *white = new Material(Vector3f(0.725f, 0.71f, 0.68f));

    Material *light = new Material(Vector3f(0.65f), (8.0f * Vector3f(0.747f + 0.058f, 0.747f + 0.258f, 0.747f) + 15.6f * Vector3f(0.740f + 0.287f, 0.740f + 0.160f, 0.740f) + 18.4f * Vector3f(0.737f + 0.642f, 0.737f + 0.159f, 0.737f)));

    MeshTriangle *floor = new MeshTriangle(path + "cornellbox/floor.obj", white);
    MeshTriangle *tallbox = new MeshTriangle(path + "cornellbox/tallbox.obj", white);
    MeshTriangle *shortbox = new MeshTriangle(path + "cornellbox/shortbox.obj", white);
    // MeshTriangle *spotlight = new MeshTriangle(path + "cornellbox/light.obj", &spotlightMaterial);
    MeshTriangle *left = new MeshTriangle(path + "cornellbox/left.obj", red);
    MeshTriangle *right = new MeshTriangle(path + "cornellbox/right.obj", green);
    MeshTriangle *light_ = new MeshTriangle(path + "cornellbox/light.obj", light);
    this->Add(floor);
    this->Add(shortbox);
    this->Add(tallbox);
    this->Add(left);
    // this->Add(spotlight);
    this->Add(right);
    this->Add(light_);
}

#endif // CUDASHADER_SCENE_H