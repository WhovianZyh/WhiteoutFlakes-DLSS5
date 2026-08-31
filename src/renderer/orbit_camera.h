#pragma once
#include "camera.h"
#include "render_settings.h"

namespace whiteout::flakes::renderer {

// Orbit speed in radians per second for each mode.
inline f32 OrbitSpeedForMode(RenderSettings::OrbitMode m) {
    switch (m) {
    case RenderSettings::OrbitMode::Slow: return 0.12f;   // ~52s per revolution
    case RenderSettings::OrbitMode::Medium: return 0.30f; // ~21s
    case RenderSettings::OrbitMode::Fast: return 0.65f;   // ~9.6s
    case RenderSettings::OrbitMode::Manual:
    default: return 0.0f;
    }
}

// Apply orbit update to camera (camera-orbit axis). No-op if manual or locked.
inline void UpdateOrbitCamera(Camera& cam, f32 dt, RenderSettings::OrbitMode mode,
                              RenderSettings::OrbitAxisMode axisMode) {
    if (mode == RenderSettings::OrbitMode::Manual) return;
    if (axisMode != RenderSettings::OrbitAxisMode::CameraOrbit) return;
    if (cam.GetMode() != Camera::Mode::Orbital) return;
    const f32 speed = OrbitSpeedForMode(mode);
    if (speed == 0.0f || dt <= 0.0f) return;
    f32 yaw = cam.GetYaw();
    yaw += speed * dt;
    // Keep yaw in [-pi, pi] to avoid float drift.
    constexpr f32 kTwoPi = 6.28318530718f;
    if (yaw > 3.14159265f) yaw -= kTwoPi;
    if (yaw < -3.14159265f) yaw += kTwoPi;
    cam.SetYaw(yaw);
}

} // namespace whiteout::flakes::renderer
