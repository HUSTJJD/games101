#include "BVH.hpp"
#include <algorithm>
#include <cassert>
#include <map>
BVHAccel::BVHAccel(std::vector<Object *> p, int maxPrimsInNode,
                   SplitMethod splitMethod)
    : maxPrimsInNode(std::min(255, maxPrimsInNode)), splitMethod(splitMethod),
      primitives(std::move(p))
{
    time_t start, stop;
    time(&start);
    if (primitives.empty())
        return;

    root = recursiveBuild(primitives);

    time(&stop);
    double diff = difftime(stop, start);
    int hrs = (int)diff / 3600;
    int mins = ((int)diff / 60) - (hrs * 60);
    int secs = (int)diff - (hrs * 3600) - (mins * 60);

    printf(
        "\rBVH Generation complete: \nTime Taken: %i hrs, %i mins, %i secs\n\n",
        hrs, mins, secs);
}

BVHBuildNode *BVHAccel::recursiveBuild(std::vector<Object *> objects)
{
    BVHBuildNode *node = new BVHBuildNode();

    // Compute bounds of all primitives in BVH node
    Bounds3 bounds;
    for (int i = 0; i < objects.size(); ++i)
        bounds = Union(bounds, objects[i]->getBounds());
    if (objects.size() == 1)
    {
        // Create leaf _BVHBuildNode_
        node->bounds = objects[0]->getBounds();
        node->object = objects[0];
        node->left = nullptr;
        node->right = nullptr;
        return node;
    }
    else if (objects.size() == 2)
    {
        node->left = recursiveBuild(std::vector{objects[0]});
        node->right = recursiveBuild(std::vector{objects[1]});

        node->bounds = Union(node->left->bounds, node->right->bounds);
        return node;
    }
    else
    {
        Bounds3 centroidBounds;
        //算出最大的包围盒（通用的，因为两个方法都要用到）
        for (int i = 0; i < objects.size(); ++i)
            centroidBounds =
                Union(centroidBounds, objects[i]->getBounds().Centroid());
        std::vector<Object *> leftshapes;
        std::vector<Object *> rightshapes;
        int dim = centroidBounds.maxExtent();
        switch (splitMethod) //这里注意了在BVH.h里面有个枚举类，构造函数中的初始将决定是普通方法，还是SAH
        {
        case SplitMethod::NAIVE:
        {
            switch (dim)
            {
            case 0:
                std::sort(objects.begin(), objects.end(), [](auto f1, auto f2)
                          { return f1->getBounds().Centroid().x <
                                   f2->getBounds().Centroid().x; });
                break;
            case 1:
                std::sort(objects.begin(), objects.end(), [](auto f1, auto f2)
                          { return f1->getBounds().Centroid().y <
                                   f2->getBounds().Centroid().y; });
                break;
            case 2:
                std::sort(objects.begin(), objects.end(), [](auto f1, auto f2)
                          { return f1->getBounds().Centroid().z <
                                   f2->getBounds().Centroid().z; });
                break;
            }

            auto beginning = objects.begin();
            auto middling = objects.begin() + (objects.size() / 2);
            auto ending = objects.end();

            auto leftshapes = std::vector<Object *>(beginning, middling);
            auto rightshapes = std::vector<Object *>(middling, ending);

            assert(objects.size() == (leftshapes.size() + rightshapes.size()));

            node->left = recursiveBuild(leftshapes);
            node->right = recursiveBuild(rightshapes);

            node->bounds = Union(node->left->bounds, node->right->bounds);
        }
        break;
        case SplitMethod::SAH:
        {
            float SN = centroidBounds.SurfaceArea();
            int B = 10;
            int mincostIndex = 0;
            float minCost = std::numeric_limits<float>::infinity(); //最小花费

            switch (dim) //直接在跨度最大的坐标轴上开始我们的算法
            {
            case 0:
                std::sort(objects.begin(), objects.end(), [](auto f1, auto f2)
                          { return f1->getBounds().Centroid().x <
                                   f2->getBounds().Centroid().x; });
                break;
            case 1:
                std::sort(objects.begin(), objects.end(), [](auto f1, auto f2)
                          { return f1->getBounds().Centroid().y <
                                   f2->getBounds().Centroid().y; });
                break;
            case 2:
                std::sort(objects.begin(), objects.end(), [](auto f1, auto f2)
                          { return f1->getBounds().Centroid().z <
                                   f2->getBounds().Centroid().z; });
                break;
            }
            for (int i = 1; i < B; i++)
            {
                auto beginning = objects.begin();
                auto middling = objects.begin() + (objects.size() * i / B);
                auto ending = objects.end();
                auto leftshapes = std::vector<Object *>(beginning, middling);
                auto rightshapes = std::vector<Object *>(middling, ending);
                //求左右包围盒:
                Bounds3 leftBounds, rightBounds;
                for (int k = 0; k < leftshapes.size(); ++k)
                    leftBounds = Union(leftBounds, leftshapes[k]->getBounds().Centroid());
                for (int k = 0; k < rightshapes.size(); ++k)
                    rightBounds = Union(rightBounds, rightshapes[k]->getBounds().Centroid());
                float SA = leftBounds.SurfaceArea();                                          // SA
                float SB = rightBounds.SurfaceArea();                                         // SB
                float cost = 0.125 + (leftshapes.size() * SA + rightshapes.size() * SB) / SN; //计算花费
                if (cost < minCost)                                                           //如果花费更小，记录当前坐标值
                {
                    minCost = cost;
                    mincostIndex = i;
                }
            }
            //找到mincostIndex后的操作等同于BVH
            auto beginning = objects.begin();
            auto middling = objects.begin() + (objects.size() * mincostIndex / B); //划分点选为当前最优桶的位置
            auto ending = objects.end();
            auto leftshapes = std::vector<Object *>(beginning, middling); //数组切分
            auto rightshapes = std::vector<Object *>(middling, ending);

            assert(objects.size() == (leftshapes.size() + rightshapes.size()));

            node->left = recursiveBuild(leftshapes); //左右开始递归
            node->right = recursiveBuild(rightshapes);
            node->bounds = Union(node->left->bounds, node->right->bounds); //返回pMin pMax构成大包围盒
        }
        }
    }
    return node;
}

Intersection BVHAccel::Intersect(const Ray &ray) const
{
    Intersection isect;
    if (!root)
        return isect;
    isect = BVHAccel::getIntersection(root, ray);
    return isect;
}

Intersection BVHAccel::getIntersection(BVHBuildNode *node, const Ray &ray) const
{
    // TODO Traverse the BVH to find intersection
    Intersection mIntersection;

    if (node == nullptr || !node->bounds.IntersectP(ray, ray.direction_inv, {0, 0, 0}))
        return mIntersection;

    if (node->left == nullptr && node->right == nullptr)
    {
        mIntersection = node->object->getIntersection(ray);
        return mIntersection;
    }

    Intersection hit1, hit2;
    hit1 = getIntersection(node->left, ray);
    hit2 = getIntersection(node->right, ray);
    mIntersection = hit1.distance <= hit2.distance ? hit1 : hit2;

    return mIntersection;
}