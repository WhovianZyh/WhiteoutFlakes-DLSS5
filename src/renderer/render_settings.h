#pragma once

// ============================================================================
// RenderSettings — application-tunable knobs that influence rendering.
//
// Pure data: atomics for things UI/render threads both touch, plain values for
// things only the render path reads. No GPU resources, no scene state, no
// pipeline behavior. The pipeline reads from here every frame; settings never
// reach back into the pipeline.
//
// IBL mode + render mode raise a one-shot dirty flag; the pipeline polls it
// and reacts on the next frame.
// ============================================================================

#include "render_target.h"             // DisplayFlags, RenderMode, LightingMode, IblMode
#include "whiteout/flakes/gfx_types.h" // gfx::GfxApi
#include "whiteout/flakes/types.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <string>

namespace whiteout::flakes::renderer {

class RenderSettings {
public:
    RenderSettings() = default;

    // ---- Display flags (what to draw) ----
    DisplayFlags GetDisplayFlags() const {
        DisplayFlags df;
        df.showGrid = showGrid_;
        df.showParticles = showParticles_;
        df.showRibbons = showRibbons_;
        df.showCollisions = showCollisions_;
        df.showLights = showLights_;
        df.showEvents = showEvents_;
        df.renderMode = renderMode_;
        return df;
    }
    void SetDisplayFlags(const DisplayFlags& f) {
        showGrid_ = f.showGrid;
        showParticles_ = f.showParticles;
        showRibbons_ = f.showRibbons;
        showCollisions_ = f.showCollisions;
        showLights_ = f.showLights;
        showEvents_ = f.showEvents;
        SetRenderMode(f.renderMode);
    }

    bool ShowGrid() const {
        return showGrid_;
    }
    bool ShowParticles() const {
        return showParticles_;
    }
    bool ShowRibbons() const {
        return showRibbons_;
    }
    bool ShowCollisions() const {
        return showCollisions_;
    }
    bool ShowLights() const {
        return showLights_;
    }
    bool ShowEvents() const {
        return showEvents_;
    }

    // ---- Render mode (HD vs SD) ----
    // App sets the mode based on what it loaded; pipeline polls the dirty
    // flag to know when to rebuild PSOs / IBL state.
    RenderMode GetRenderMode() const {
        return renderMode_;
    }
    void SetRenderMode(RenderMode m) {
        if (renderMode_ != m) {
            renderMode_ = m;
            renderModeDirty_ = true;
        }
    }
    bool ConsumeRenderModeDirty() {
        return renderModeDirty_.exchange(false);
    }

    // Route the SD shading path through the HDR scene target + tonemap
    // instead of straight onto the LDR swap chain. Keeps authentic SD
    // shading (single RTV, no G-buffer/PBR) but lets additive / team-color
    // geosets accumulate in float and roll off through the tonemap rather
    // than clipping to opaque white. Off = classic direct-to-swap-chain SD
    // (pixel-identical to before). HD mode always renders to HDR and ignores
    // this flag.
    bool SceneHdrInSd() const {
        return sceneHdrInSd_.load();
    }
    void SetSceneHdrInSd(bool on) {
        sceneHdrInSd_.store(on);
    }

    // ---- Debug visualization ----
    i32 HdDebugMode() const {
        return hdDebugMode_.load();
    }
    void SetHdDebugMode(i32 m) {
        hdDebugMode_.store(m);
    }

    i32 LodOverride() const {
        return lodOverride_.load();
    }
    void SetLodOverride(i32 l) {
        lodOverride_.store(l);
    }

    // ---- Ambient occlusion (GTAO) ----
    // Gates the HD-mode GTAO pass between the G-buffer close and the
    // tonemap. Off skips both the main and apply draws — the AO buffer
    // is allocated either way (cheap, 1 byte per pixel) so the flip is
    // free of resize work. SD mode ignores this flag.
    bool AoEnabled() const {
        return aoEnabled_.load();
    }
    void SetAoEnabled(bool on) {
        aoEnabled_.store(on);
    }

    // GTAO quality preset — index into gtao::Quality (Low=0, Medium=1,
    // High=2). The pipeline forwards this into the service each frame
    // so the user can A/B presets without a restart.
    u32 AoQuality() const {
        return aoQuality_.load();
    }
    void SetAoQuality(u32 q) {
        aoQuality_.store(q);
    }

    // Bent-normal IBL boost strength. 0 = pass disabled (no-op). Tiny
    // values (~0.05) recover some of the indirect-light energy lost when
    // GTAO darkens cavities; bigger values overdrive the model. Stored
    // as raw bits in an atomic<u32> because std::atomic<f32> isn't a
    // standard specialisation pre-C++20 across all our toolchains.
    f32 AoBentBoost() const {
        const u32 bits = aoBentBoost_.load();
        f32 v;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    }
    void SetAoBentBoost(f32 v) {
        u32 bits;
        std::memcpy(&bits, &v, sizeof(bits));
        aoBentBoost_.store(bits);
    }

    // ---- Bloom (HD-only) ----
    // Master enable. Off ⇒ PostProcessService::RunBloom is a no-op.
    bool BloomEnabled() const {
        return bloomEnabled_.load();
    }
    void SetBloomEnabled(bool on) {
        bloomEnabled_.store(on);
    }
    // Threshold + intensity + saturation float bits stored in atomic<u32>
    // — same trick as AoBentBoost (std::atomic<f32> isn't portable).
    f32 BloomThreshold() const {
        const u32 bits = bloomThreshold_.load();
        f32 v;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    }
    void SetBloomThreshold(f32 v) {
        u32 bits;
        std::memcpy(&bits, &v, sizeof(bits));
        bloomThreshold_.store(bits);
    }
    f32 BloomIntensity() const {
        const u32 bits = bloomIntensity_.load();
        f32 v;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    }
    void SetBloomIntensity(f32 v) {
        u32 bits;
        std::memcpy(&bits, &v, sizeof(bits));
        bloomIntensity_.store(bits);
    }
    f32 BloomSaturation() const {
        const u32 bits = bloomSaturation_.load();
        f32 v;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    }
    void SetBloomSaturation(f32 v) {
        u32 bits;
        std::memcpy(&bits, &v, sizeof(bits));
        bloomSaturation_.store(bits);
    }

    // ---- FXAA (HD-only) ----
    // Master enable. Off ⇒ PostProcessService::RunFxaa is a no-op. Runs on
    // hdrColor between bloom and tonemap; no compute/UAV needed.
    bool FxaaEnabled() const {
        return fxaaEnabled_.load();
    }
    void SetFxaaEnabled(bool on) {
        fxaaEnabled_.store(on);
    }

    // ---- SMAA 1x (HD-only) ----
    // Same position as FXAA; the viewer UI treats the two antialiasing
    // toggles as mutually exclusive (enabling one disables the other).
    bool SmaaEnabled() const {
        return smaaEnabled_.load();
    }
    void SetSmaaEnabled(bool on) {
        smaaEnabled_.store(on);
    }

    // ---- CAS sharpening (HD-only, after bloom/AA, before tonemap) ----
    bool CasEnabled() const {
        return casEnabled_.load();
    }
    void SetCasEnabled(bool on) {
        casEnabled_.store(on);
    }
    f32 CasSharpness() const {
        return loadF32(casSharpness_);
    }
    void SetCasSharpness(f32 v) {
        storeF32(casSharpness_, std::clamp(v, 0.0f, 1.0f));
    }

    // ---- Depth of field (HD-only) ----
    // Master enable. Off (and a focal distance of 0) ⇒ DofService::Run is a
    // no-op. Mirrors WC3's per-camera GetDepthOfFieldEnabled gate.
    bool DofEnabled() const {
        return dofEnabled_.load();
    }
    void SetDofEnabled(bool on) {
        dofEnabled_.store(on);
    }
    // Float bits in atomic<u32>, same trick as the bloom knobs above.
    //   FocusDistance — linear view-Z (scene units) kept in focus; 0 disables.
    //   FocusScale    — circle-of-confusion ramp (WC3 CameraSetDepthOfFieldScale).
    //   MaxBlurSize   — max gather radius in px (WC3 default 10).
    //   RadiusScale   — spiral ring spacing / sample density (WC3 default 1).
    f32 DofFocusDistance() const {
        return loadF32(dofFocusDistance_);
    }
    void SetDofFocusDistance(f32 v) {
        storeF32(dofFocusDistance_, v);
    }
    f32 DofFocusScale() const {
        return loadF32(dofFocusScale_);
    }
    void SetDofFocusScale(f32 v) {
        storeF32(dofFocusScale_, v);
    }
    f32 DofMaxBlurSize() const {
        return loadF32(dofMaxBlurSize_);
    }
    void SetDofMaxBlurSize(f32 v) {
        storeF32(dofMaxBlurSize_, v);
    }
    f32 DofRadiusScale() const {
        return loadF32(dofRadiusScale_);
    }
    void SetDofRadiusScale(f32 v) {
        storeF32(dofRadiusScale_, v);
    }
    bool DofFarFieldOnly() const {
        return dofFarFieldOnly_.load();
    }
    void SetDofFarFieldOnly(bool on) {
        dofFarFieldOnly_.store(on);
    }

    // ---- Lighting / clear color ----
    LightingMode GetLightingMode() const {
        return static_cast<LightingMode>(lightingMode_.load());
    }
    void SetLightingMode(LightingMode m) {
        lightingMode_.store(static_cast<u8>(m));
    }

    u32 BackgroundColorRaw() const {
        return backgroundColor_.load();
    }
    void SetBackgroundColor(u8 r, u8 g, u8 b) {
        backgroundColor_.store(u32(r) | (u32(g) << 8) | (u32(b) << 16));
    }

    // ---- IBL ----
    // Pipeline polls ConsumeIblModeDirty() each frame; if set, it reloads
    // probe textures based on GetIblMode().
    IblMode GetIblMode() const {
        return iblMode_;
    }
    void SetIblMode(IblMode m) {
        if (iblMode_ != m) {
            iblMode_ = m;
            iblModeDirty_ = true;
        } else {
            iblModeDirty_ = true;
        }
    }
    bool ConsumeIblModeDirty() {
        return iblModeDirty_.exchange(false);
    }

    // ---- Tonemap ----
    f32 GetTonemapExposure() const {
        return tonemapExposure_;
    }
    void SetTonemapExposure(f32 e) {
        tonemapExposure_ = e;
    }

    // ---- Graphics debug ----
    // Mirrored as an init-time flag: read once at device creation in
    // RenderPipeline::InitDevice and passed into gfx::CreateDevice,
    // which routes it to the per-backend validation layer (Vulkan
    // VK_LAYER_KHRONOS_validation, d3d11 DEBUG flag, d3d12 debug layer
    // + InfoQueue break-on-error). Changing it mid-run has no effect
    // until the next process start — the host persists it to .ini so
    // the next launch picks the new value up.
    bool GraphicsDebug() const {
        return graphicsDebug_;
    }
    void SetGraphicsDebug(bool v) {
        graphicsDebug_ = v;
    }

    // ---- Default GFX backend ----
    // The host (basic_viewer) reads this when no `--backend` argument
    // was given on the command line. RenderSettings stores it so the
    // existing INI save/load + Settings-window infrastructure can be
    // reused; the renderer itself never inspects it (its backend is
    // already fixed by the time RenderPipeline::InitDevice runs).
    gfx::GfxApi DefaultBackend() const {
        return defaultBackend_;
    }
    void SetDefaultBackend(gfx::GfxApi b) {
        defaultBackend_ = b;
    }

    // ---- Orbit camera (Viewer host-driven, Task 1.3) ----
    enum class OrbitMode : i32 { Manual = 0, Slow = 1, Medium = 2, Fast = 3 };
    enum class OrbitAxisMode : i32 { CameraOrbit = 0, ModelRotate = 1 };
    OrbitMode GetOrbitMode() const {
        return static_cast<OrbitMode>(orbitMode_.load());
    }
    void SetOrbitMode(OrbitMode m) {
        orbitMode_.store(static_cast<i32>(m));
    }
    OrbitAxisMode GetOrbitAxisMode() const {
        return static_cast<OrbitAxisMode>(orbitAxisMode_.load());
    }
    void SetOrbitAxisMode(OrbitAxisMode m) {
        orbitAxisMode_.store(static_cast<i32>(m));
    }

    // ---- PN-Triangle geometry enhancement (Task 2) ----
    enum class PnMode : i32 { Off = 0, X2 = 1, X4 = 2, X8 = 3, Adaptive = 4 };
    PnMode GetPnMode() const {
        return static_cast<PnMode>(pnMode_.load());
    }
    void SetPnMode(PnMode m) {
        pnMode_.store(static_cast<i32>(m));
    }
    f32 PnTessFactor() const {
        switch (GetPnMode()) {
        case PnMode::X2: return 2.0f;
        case PnMode::X4: return 4.0f;
        case PnMode::X8: return 8.0f;
        case PnMode::Adaptive: return 4.0f;
        default: return 1.0f;
        }
    }
    // Adaptive: distance in world units (camera eye -> target). 600/1200/2400 tiers.
    f32 PnAdaptiveFactor(f32 dist) const {
        if (GetPnMode() != PnMode::Adaptive)
            return PnTessFactor();
        if (dist < 600.0f) return 8.0f;
        if (dist < 1200.0f) return 4.0f;
        if (dist < 2400.0f) return 2.0f;
        return 1.0f;
    }
    f32 PnCreaseThreshold() const {
        return loadF32(pnCreaseThreshold_);
    }
    void SetPnCreaseThreshold(f32 v) {
        storeF32(pnCreaseThreshold_, v);
    }

    // ---- Motion vector & Neural rendering (Tasks 3/1) ----
    enum class MvSource : i32 { OpticalFlow = 0, RendererTrueMV = 1 };
    MvSource GetMvSource() const {
        return static_cast<MvSource>(mvSource_.load());
    }
    void SetMvSource(MvSource s) {
        mvSource_.store(static_cast<i32>(s));
    }
    bool VelocityDebug() const {
        return velocityDebug_.load();
    }
    void SetVelocityDebug(bool v) {
        velocityDebug_.store(v);
    }
    bool VelocityFlipX() const {
        return velocityFlipX_.load();
    }
    void SetVelocityFlipX(bool v) {
        velocityFlipX_.store(v);
    }
    bool VelocityFlipY() const {
        return velocityFlipY_.load();
    }
    void SetVelocityFlipY(bool v) {
        velocityFlipY_.store(v);
    }
    bool NeuralRenderingEnabled() const {
        return neuralRenderingEnabled_.load();
    }
    void SetNeuralRenderingEnabled(bool v) {
        neuralRenderingEnabled_.store(v);
    }

    // ---- Frame rate limiter ----
    int GetMaxFps() const {
        return maxFps_.load();
    }
    void SetMaxFps(int fps) {
        maxFps_.store(fps);
    }

    // ---- Anisotropic filtering (material samplers) ----
    // 1=off, 2/4/8/16x. Applied to SamplerAssetManager's WrapVariant/LinearWrap.
    u32 GetAnisotropy() const {
        return anisotropy_.load();
    }
    void SetAnisotropy(u32 v) {
        if (v != 1 && v != 2 && v != 4 && v != 8 && v != 16)
            v = 4;
        anisotropy_.store(v);
    }
    f32 GetMipBias() const { return loadF32(mipBias_); }
    void SetMipBias(f32 v) { storeF32(mipBias_, std::clamp(v, -1.0f, 0.5f)); }

    // ---- Preferred GFX device ----
    // Exact-match name of the physical adapter the host wants the
    // selected backend to open (compared verbatim against the names
    // returned by gfx::EnumerateDevices). Empty (the default) means
    // "let the backend pick — highest VRAM / discrete over integrated".
    // Plumbed in InitDevice via gfx::SetPreferredDevice before
    // gfx::CreateDevice. Takes effect on the next process launch.
    const std::string& PreferredDevice() const {
        return preferredDevice_;
    }
    void SetPreferredDevice(std::string name) {
        preferredDevice_ = std::move(name);
    }

private:
    // Bit-cast helpers for the float-in-atomic<u32> knobs (atomic<f32> isn't
    // portable). Used by the depth-of-field accessors above.
    static f32 loadF32(const std::atomic<u32>& a) {
        const u32 bits = a.load();
        f32 v;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    }
    static void storeF32(std::atomic<u32>& a, f32 v) {
        u32 bits;
        std::memcpy(&bits, &v, sizeof(bits));
        a.store(bits);
    }

    // Display flags — plain bools; readers tolerate single-byte tearing.
    bool showGrid_ = false;
    bool showParticles_ = true;
    bool showRibbons_ = true;
    bool showCollisions_ = false;
    bool showLights_ = false;
    bool showEvents_ = true;

    // Render mode + dirty flag.
    RenderMode renderMode_ = RenderMode::SD;
    std::atomic<bool> renderModeDirty_{false};
    std::atomic<bool> sceneHdrInSd_{false};

    // Debug + LOD.
    std::atomic<i32> hdDebugMode_{0};
    std::atomic<i32> lodOverride_{0};

    // Ambient occlusion (HD-mode GTAO). On by default — the user can
    // disable from the settings menu if they don't want the pass.
    std::atomic<bool> aoEnabled_{true};

    // Quality preset (Low=0, Medium=1, High=2). High by default — the
    // extra slice/step taps comfortably fit the budget on every backend
    // we ship, and the lower presets exist mainly as a fallback.
    std::atomic<u32> aoQuality_{2};

    // Bent-normal IBL boost (float bits in u32 — atomic<f32> isn't
    // portable). 0 disables the boost pass.
    std::atomic<u32> aoBentBoost_{0};

    // Bloom — defaults match the engine's RegisterBloom (BL_BLOOM_D=0
    // off, threshold=1.0, intensity=1.25, saturation=1.0). Floats stored
    // as raw bits in an atomic<u32>; helper bit-casts are below.
    std::atomic<bool> bloomEnabled_{false};
    std::atomic<u32> bloomThreshold_{0x3F800000u}; // 1.0f
    std::atomic<u32> bloomIntensity_{0x3FA00000u}; // 1.25f
    std::atomic<u32> bloomSaturation_{0x3F800000u}; // 1.0f

    // FXAA — off by default (pure look preference, toggled from the View
    // menu and persisted via the ini).
    std::atomic<bool> fxaaEnabled_{false};

    // SMAA 1x — off by default, same UI/ini story as FXAA.
    std::atomic<bool> smaaEnabled_{false};

    // CAS — off by default (user opts in; pairs well with AF + PN)
    std::atomic<bool> casEnabled_{false};
    std::atomic<u32> casSharpness_{0x3ecccccd}; // 0.4f

    // Depth of field — off by default (the host supplies a focal distance).
    // Defaults mirror WC3: maxBlurSize=10, radiusScale=1, focusScale=1.
    std::atomic<bool> dofEnabled_{false};
    std::atomic<u32> dofFocusDistance_{0};            // 0.0f — disables the pass
    std::atomic<u32> dofFocusScale_{0x3F800000u};     // 1.0f
    std::atomic<u32> dofMaxBlurSize_{0x41200000u};    // 10.0f
    std::atomic<u32> dofRadiusScale_{0x3F800000u};    // 1.0f
    std::atomic<bool> dofFarFieldOnly_{false};

    // Lighting + clear color.
    std::atomic<u8> lightingMode_{static_cast<u8>(LightingMode::InGame)};
    std::atomic<u32> backgroundColor_{0x00453A35u};

    // IBL. The world default has to be the day/night pair, not the portrait
    // probe: `CWorldFrameWar3` loads `[DayEnvironmentMap]`/`[NightEnvironmentMap]`
    // for the map's tileset, while `[PortraitEnvironmentMap]` belongs to
    // `CPortraitButton` alone. It also matters more than it looks — the HD DNC
    // rigs carry `ambientIntensity = 0`, so on the plain-HD shader this probe
    // is the entire ambient term, and the portrait probe is ~30% dimmer than
    // Lordaeron Summer's day map (mean luma 0.17 vs 0.25) and never varies
    // with time of day.
    IblMode iblMode_ = IblMode::DayNight;
    std::atomic<bool> iblModeDirty_{true}; // pipeline does an initial apply

    // Tonemap exposure.
    f32 tonemapExposure_ = 1.0f;

    // Graphics-debug (validation layers). Off by default — turning it
    // on costs frame time and noise. Set before InitDevice runs.
    bool graphicsDebug_ = false;

    // Host-only: default backend when --backend is not on the command
    // line. D3D12 matches the long-standing test_main default on
    // Windows; macOS prefers Metal (D3D12 isn't available there);
    // every other platform falls back to Vulkan (which is the only
    // backend the gfx_factory builds elsewhere).
#if defined(_WIN32)
    gfx::GfxApi defaultBackend_ = gfx::GfxApi::D3D12;
#elif defined(__APPLE__)
    gfx::GfxApi defaultBackend_ = gfx::GfxApi::Metal;
#else
    gfx::GfxApi defaultBackend_ = gfx::GfxApi::Vulkan;
#endif

    // Host-only: preferred physical device name. Empty = default pick.
    std::string preferredDevice_;

    // Orbit camera state (host-driven, persisted via INI).
    std::atomic<i32> orbitMode_{static_cast<i32>(OrbitMode::Manual)};
    std::atomic<i32> orbitAxisMode_{static_cast<i32>(OrbitAxisMode::CameraOrbit)};

    // PN-Triangle state (Task 2).
    std::atomic<i32> pnMode_{static_cast<i32>(PnMode::Off)};
    std::atomic<u32> pnCreaseThreshold_{0x42700000u}; // 60.0f degrees

    // Motion vector + neural rendering (Tasks 3/1).
    std::atomic<i32> mvSource_{static_cast<i32>(MvSource::OpticalFlow)};
    std::atomic<bool> velocityDebug_{false};
    std::atomic<bool> velocityFlipX_{false};
    std::atomic<bool> velocityFlipY_{true}; // Y flip commonly needed for UV vs NDC
    std::atomic<bool> neuralRenderingEnabled_{false};

    // Frame rate limiter (0 = unlimited, default 120 as requested)
    std::atomic<int> maxFps_{120};

    // Anisotropic filtering, default 4x (best cost/quality for WC3 low-poly)
    std::atomic<u32> anisotropy_{4};
    std::atomic<u32> mipBias_{0xbd99999a}; // -0.3f
};

} // namespace whiteout::flakes::renderer
