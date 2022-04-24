#ifndef CAMERA_H
#define CAMERA_H

#include <glad/glad.h>

#include <glm/matrix.hpp>
#include <vector>

#include "../common/Setting.h"
#include "../common/float3Extension.h"
// Defines several possible options for camera movement. Used as abstraction to
// stay away from window-system specific input methods
enum Camera_Movement { FORWARD, BACKWARD, LEFT, RIGHT };

float radians(const float& angle) const {}

// An abstract camera class that processes input and calculates the
// corresponding Euler Angles, Vectors and Matrices for use in OpenGL
class Camera {
 public:
  // camera Attributes
  float3 Position;
  float3 Front;
  float3 WorldUp;

  float Yaw;
  float Pitch;
  // camera options
  float MovementSpeed;
  float MouseSensitivity;
  float Zoom;

  // euler Angles
  float3 Up;
  float3 Right;

  // constructor with vectors
  Camera(float3 position = make_float3(0.0f, 0.0f, 0.0f),
         float3 up = make_float3(0.0f, 1.0f, 0.0f), float yaw = Setting::YAW,
         float pitch = Setting::PITCH)
      : Position(position),
        Front(make_float3(0.0f, 0.0f, -1.0f)),
        WorldUp(up),
        Yaw(yaw),
        Pitch(pitch),
        MovementSpeed(Setting::SPEED),
        MouseSensitivity(Setting::SENSITIVITY),
        Zoom(Setting::ZOOM) {
    updateCameraVectors();
  }

  // returns the view matrix calculated using Euler Angles and the LookAt Matrix
  glm::mat4 GetViewMatrix() {
    return glm::lookAt(Position, Position + Front, Up);
  }

  // processes input received from any keyboard-like input system. Accepts input
  // parameter in the form of camera defined ENUM (to abstract it from windowing
  // systems)
  void ProcessKeyboard(Camera_Movement direction, float deltaTime) {
    float velocity = MovementSpeed * deltaTime;
    if (direction == FORWARD) Position += Front * velocity;
    if (direction == BACKWARD) Position -= Front * velocity;
    if (direction == LEFT) Position -= Right * velocity;
    if (direction == RIGHT) Position += Right * velocity;
  }

  // processes input received from a mouse input system. Expects the offset
  // value in both the x and y direction.
  void ProcessMouseMovement(float xoffset, float yoffset,
                            GLboolean constrainPitch = true) {
    xoffset *= MouseSensitivity;
    yoffset *= MouseSensitivity;

    Yaw += xoffset;
    Pitch += yoffset;

    // make sure that when pitch is out of bounds, screen doesn't get flipped
    if (constrainPitch) {
      if (Pitch > 89.0f) Pitch = 89.0f;
      if (Pitch < -89.0f) Pitch = -89.0f;
    }

    // update Front, Right and Up Vectors using the updated Euler angles
    updateCameraVectors();
  }

  // processes input received from a mouse scroll-wheel event. Only requires
  // input on the vertical wheel-axis
  void ProcessMouseScroll(float yoffset) {
    Zoom -= (float)yoffset;
    if (Zoom < 1.0f) Zoom = 1.0f;
    if (Zoom > 45.0f) Zoom = 45.0f;
  }

 private:
  // calculates the front vector from the Camera's (updated) Euler Angles
  void updateCameraVectors() {
    // calculate the new Front vector
    float3 front;
    front.x = cos(Yaw) * cos(Pitch);
    front.y = sin(Pitch);
    front.z = sin(Yaw) * cos(Pitch);
    Front = Normalize(front);
    // also re-calculate the Right and Up vector
    Right = Normalize(
        Cross(Front, WorldUp));  // normalize the vectors, because their length
                                 // gets closer to 0 the more you look up or
                                 // down which results in slower movement.
    Up = Normalize(Cross(Right, Front));
  }
};
#endif