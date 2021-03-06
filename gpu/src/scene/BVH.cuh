#ifndef GPU_BVH_H
#define GPU_BVH_H

struct Bounds3;
struct Triangle;
struct BVH;
struct Ray;

int box_x_compare(const void* a, const void* b);
int box_y_compare(const void* a, const void* b);
int box_z_compare(const void* a, const void* b);
BVH* BuildBVH(Triangle* tri);
BVH* BuildBVH(Triangle** list, int n);
BVH* ToDevice(BVH*);
__host__ __device__ void Print(BVH* bvh, bool root = false);
struct BVH {
  bool tri;
  Triangle* triangle;
  Bounds3* aabb;
  BVH* left;
  BVH* right;
  BVH() {}
};

// struct BVHBuildNode;
// // BVHAccel Forward Declarations
// struct BVHPrimitiveInfo;

// // BVHAccel Declarations
// inline int leafNodes, totalLeafNodes, totalPrimitives, interiorNodes;
// class BVHAccel {
//  public:
//   // BVHAccel Public Types
//   enum class SplitMethod { NAIVE, SAH };

//   // BVHAccel Public Methods
//   BVHAccel(std::vector<Object *> p, int maxPrimsInNode = 1,
//            SplitMethod splitMethod = SplitMethod::NAIVE);
//   Bounds3 WorldBound() const;

//   Intersection Intersect(const Ray &ray) const;
//   Intersection getIntersection(BVHBuildNode *node, const Ray &ray) const;
//   bool IntersectP(const Ray &ray) const;
//   BVHBuildNode *root;

//   // BVHAccel Private Methods
//   BVHBuildNode *recursiveBuild(std::vector<Object *> objects);

//   // BVHAccel Private Data
//   const int maxPrimsInNode;
//   const SplitMethod splitMethod;
//   std::vector<Object *> primitives;

//   void getSample(BVHBuildNode *node, float p, Intersection &pos, float &pdf);
//   void Sample(Intersection &pos, float &pdf);
// };

// struct BVHBuildNode {
//   Bounds3 bounds;
//   BVHBuildNode *left;
//   BVHBuildNode *right;
//   Object *object;
//   float area;

//  public:
//   int splitAxis = 0, firstPrimOffset = 0, nPrimitives = 0;
//   // BVHBuildNode Public Methods
//   BVHBuildNode() {
//     bounds = Bounds3();
//     left = nullptr;
//     right = nullptr;
//     object = nullptr;
//   }
// };

// BVHAccel::BVHAccel(std::vector<Object *> p, int maxPrimsInNode,
//                    SplitMethod splitMethod)
//     : maxPrimsInNode(std::min(255, maxPrimsInNode)),
//       splitMethod(splitMethod),
//       primitives(std::move(p)) {
//   time_t start, stop;
//   time(&start);
//   if (primitives.empty()) return;

//   root = recursiveBuild(primitives);

//   time(&stop);
//   double diff = difftime(stop, start);
//   int hrs = (int)diff / 3600;
//   int mins = ((int)diff / 60) - (hrs * 60);
//   int secs = (int)diff - (hrs * 3600) - (mins * 60);

//   printf(
//       "\rBVH Generation complete: \nTime Taken: %i hrs, %i mins, %i
//       secs\n\n", hrs, mins, secs);
// }

// BVHBuildNode *BVHAccel::recursiveBuild(std::vector<Object *> objects) {
//   BVHBuildNode *node = new BVHBuildNode();

//   // Compute bounds of all primitives in BVH node
//   Bounds3 bounds;
//   for (int i = 0; i < objects.size(); ++i)
//     bounds = Union(bounds, objects[i]->getBounds());
//   if (objects.size() == 1) {
//     // Create leaf _BVHBuildNode_
//     node->bounds = objects[0]->getBounds();
//     node->object = objects[0];
//     node->left = nullptr;
//     node->right = nullptr;
//     node->area = objects[0]->getArea();
//     return node;
//   } else if (objects.size() == 2) {
//     node->left = recursiveBuild(std::vector{objects[0]});
//     node->right = recursiveBuild(std::vector{objects[1]});

//     node->bounds = Union(node->left->bounds, node->right->bounds);
//     node->area = node->left->area + node->right->area;
//     return node;
//   } else {
//     Bounds3 centroidBounds;
//     for (int i = 0; i < objects.size(); ++i)
//       centroidBounds =
//           Union(centroidBounds, objects[i]->getBounds().Centroid());
//     int dim = centroidBounds.maxExtent();
//     switch (dim) {
//       case 0:
//         std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
//           return f1->getBounds().Centroid().x < f2->getBounds().Centroid().x;
//         });
//         break;
//       case 1:
//         std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
//           return f1->getBounds().Centroid().y < f2->getBounds().Centroid().y;
//         });
//         break;
//       case 2:
//         std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
//           return f1->getBounds().Centroid().z < f2->getBounds().Centroid().z;
//         });
//         break;
//     }

//     auto beginning = objects.begin();
//     auto middling = objects.begin() + (objects.size() / 2);
//     auto ending = objects.end();

//     auto leftshapes = std::vector<Object *>(beginning, middling);
//     auto rightshapes = std::vector<Object *>(middling, ending);

//     assert(objects.size() == (leftshapes.size() + rightshapes.size()));

//     node->left = recursiveBuild(leftshapes);
//     node->right = recursiveBuild(rightshapes);

//     node->bounds = Union(node->left->bounds, node->right->bounds);
//     node->area = node->left->area + node->right->area;
//   }

//   return node;
// }

// Intersection BVHAccel::Intersect(const Ray &ray) const {
//   Intersection isect;
//   if (!root) return isect;
//   isect = BVHAccel::getIntersection(root, ray);
//   return isect;
// }

// Intersection BVHAccel::getIntersection(BVHBuildNode *node,
//                                        const Ray &ray) const {
//   // TODO Traverse the BVH to find intersection
//   Intersection inter;

//   // ????????????
//   float x = ray.direction.x;
//   float y = ray.direction.y;
//   float z = ray.direction.z;
//   // ????????????????????????
//   std::array<int, 3> dirsIsNeg{int(x > 0), int(y > 0), int(z > 0)};

//   // ?????????????????????????????????????????????
//   if (node->bounds.IntersectP(ray, ray.direction_inv, dirsIsNeg) == false)
//     return inter;

//   if (node->left == nullptr && node->right == nullptr) {
//     inter = node->object->getIntersection(ray);
//     return inter;
//   }

//   // ?????????????????????????????????????????????????????????
//   auto hit1 = getIntersection(node->left, ray);
//   auto hit2 = getIntersection(node->right, ray);

//   if (hit1.distance < hit2.distance) return hit1;
//   return hit2;
// }

// void BVHAccel::getSample(BVHBuildNode *node, float p, Intersection &pos,
//                          float &pdf) {
//   if (node->left == nullptr || node->right == nullptr) {
//     node->object->Sample(pos, pdf);
//     pdf *= node->area;
//     return;
//   }
//   if (p < node->left->area)
//     getSample(node->left, p, pos, pdf);
//   else
//     getSample(node->right, p - node->left->area, pos, pdf);
// }

// void BVHAccel::Sample(Intersection &pos, float &pdf) {
//   float p = std::sqrt(get_random_float()) * root->area;
//   getSample(root, p, pos, pdf);
//   pdf /= root->area;
// }
#endif  // GPU_
