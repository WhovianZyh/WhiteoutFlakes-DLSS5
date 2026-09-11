#pragma once

#include <algorithm>
#include <functional>
#include <unordered_map>
#include "gfx/gfx.h"
#include "whiteout/flakes/types.h"

namespace whiteout::flakes::renderer::assets {

inline constexpr u32 kSamplerWrapBitsMask = 0x3;

enum class WrapMode : u32 {
    ClampClamp = 0,
    WrapClamp = 1,
    ClampWrap = 2,
    WrapWrap = 3,
};

class SamplerAssetManager {
public:
    explicit SamplerAssetManager(gfx::IGFXDevice& gfx);
    ~SamplerAssetManager();

    /// Destroy every cached sampler. Must be called while the gfx device
    /// is still alive — the destructor cannot, see its comment.
    void ReleaseGpu();

    SamplerAssetManager(const SamplerAssetManager&) = delete;
    SamplerAssetManager& operator=(const SamplerAssetManager&) = delete;

    gfx::SamplerHandle Get(const gfx::SamplerDesc& desc);

    gfx::SamplerHandle WrapVariant(u32 wrapFlags);
    gfx::SamplerHandle WrapVariant(WrapMode mode) {
        return WrapVariant(static_cast<u32>(mode));
    }

    gfx::SamplerHandle LinearWrap();

    void SetAnisotropy(u32 maxAniso) {
        anisotropy_ = std::clamp(maxAniso, 1u, 16u);
        // Power-of-two snap: 3/5/6/7 etc -> nearest pow2 for stable cache keys
        if (anisotropy_ > 1 && (anisotropy_ & (anisotropy_ - 1)) != 0) {
            // round up to next pow2: 3->4, 6->8, 12->16
            u32 p = 1;
            while (p < anisotropy_) p <<= 1;
            anisotropy_ = std::min(p, 16u);
        }
    }
    u32 Anisotropy() const noexcept { return anisotropy_; }
    void SetMipBias(f32 bias) { mipBias_ = std::clamp(bias, -1.0f, 0.5f); }
    f32 MipBias() const noexcept { return mipBias_; }

    // Hardware comparison sampler for shadow PCF. Linear filter +
    // ClampToEdge + comparison=LessEqual. The HD opaque PS sampler
    // `sd_s_shadow0..2` is declared `SamplerComparisonState` and uses
    // `SampleCmpLevelZero`, which requires a real comparison sampler.
    // Single shared handle since all three cascades use identical
    // settings; lazily created on first call.
    gfx::SamplerHandle ShadowComparison();

    usize DebugSamplerCount() const noexcept {
        return cache_.size();
    }

private:
    gfx::IGFXDevice& gfx_;
    gfx::SamplerHandle shadowComparison_ = gfx::SamplerHandle::Invalid;
    u32 anisotropy_ = 4; // default 4x for low-poly clarity; overridden by RenderSettings each frame
    f32 mipBias_ = -0.3f;

    struct DescKey {
        gfx::Filter minF;
        gfx::Filter magF;
        gfx::AddressMode aU;
        gfx::AddressMode aV;
        gfx::AddressMode aW;
        u32 maxAnisotropy = 1;
        i32 mipBiasQ = 0; // bias*100 quantized
        bool comparison = false;
        bool operator==(const DescKey&) const noexcept = default;
    };
    struct DescKeyHash {
        usize operator()(const DescKey& k) const noexcept {
            u64 h = 0;
            h |= (u64)k.minF << 0;
            h |= (u64)k.magF << 4;
            h |= (u64)k.aU << 8;
            h |= (u64)k.aV << 16;
            h |= (u64)k.aW << 24;
            h |= (u64)(k.maxAnisotropy & 0x1F) << 32;
            h |= (u64)(k.comparison ? 1 : 0) << 40;
            h ^= std::hash<i32>{}(k.mipBiasQ) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return std::hash<u64>{}(h);
        }
    };
    std::unordered_map<DescKey, gfx::SamplerHandle, DescKeyHash> cache_;
};

} // namespace whiteout::flakes::renderer::assets
