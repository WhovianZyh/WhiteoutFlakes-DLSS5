// ============================================================================
// AMD FidelityFX CAS - Contrast Adaptive Sharpening (MIT)
// Adapted from FidelityFX CAS v1.3 (c) AMD, trimmed for WhiteoutFlakes HDR path.
//
// Runs on hdrColor AFTER bloom/AA and BEFORE tonemap (see
// RenderPipeline::RenderViewport order). Like FXAA/SMAA, it operates on
// linear HDR (R11G11B10) — the tonemap downstream expects HDR range. The
// filter is relative (peak = -rcp(8*D+4*C)), so absolute luminance scaling
// is tolerated; very bright pixels are clamped to avoid halo.
//
// Binding layout mirrors fxaa.slang / gaussian_blur.slang:
//   set 0 binding 16/4  — CasCB (pixel-stage uniform)
//   set 1 binding 16/12 — source texture (hdrColor)
//   set 2 binding 16/12 — linear-clamp sampler
// ============================================================================

struct CasCB
{
    // (.x,.y)=1/w,1/h  (.z,.w)=0
    float4 rcpFrame;
    // 0.0 = off (no-op), 0.0..1.0 = sharpness. Default 0.4.
    // Mapped from user slider: sharpness = user * 1.0.
    float sharpness;
    float3 _pad;
}

#if defined(WGSL_TARGET) || defined(METAL_TARGET)
static const int WF_PS_CB_OFFSET   = 4;
static const int WF_PS_BIND_OFFSET = 12;
#else
static const int WF_PS_CB_OFFSET   = 16;
static const int WF_PS_BIND_OFFSET = 16;
#endif

[[vk::binding(WF_PS_CB_OFFSET + 0, 0)]] ConstantBuffer<CasCB> g_cb       : register(b0);
[[vk::binding(WF_PS_BIND_OFFSET + 0, 1)]] Texture2D<float4>   g_source   : register(t0);
[[vk::binding(WF_PS_BIND_OFFSET + 0, 2)]] SamplerState        g_sampler  : register(s0);

struct VsOut
{
    float4 position : SV_POSITION;
    float2 uv       : TEXCOORD0;
}

[shader("vertex")]
VsOut vsCas(uint vid : SV_VertexID)
{
    VsOut o;
    o.uv       = float2((vid << 1) & 2, vid & 2);
    o.position = float4(o.uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    return o;
}

// Luma for contrast detection — green-as-luma like FXAA (hdrColor is
// R11G11B10, no alpha). Matches FxaaLuma.
float CasLuma(float3 c) { return c.g; }

[shader("fragment")]
float4 psCas(VsOut input) : SV_Target
{
    float2 uv = input.uv;
    float sharp = g_cb.sharpness;
    // Early-out: 0 sharpening = passthrough (no texture fetch penalty beyond the center tap)
    // The branch is uniform per draw (CB), so it collapses to predicate.
    float2 rcp = g_cb.rcpFrame.xy;

    // 5-tap: center + N/S/E/W
    float3 a = g_source.SampleLevel(g_sampler, uv + float2( 0.0, -1.0) * rcp, 0.0).rgb;
    float3 b = g_source.SampleLevel(g_sampler, uv + float2(-1.0,  0.0) * rcp, 0.0).rgb;
    float3 c = g_source.SampleLevel(g_sampler, uv, 0.0).rgb;
    float3 d = g_source.SampleLevel(g_sampler, uv + float2( 1.0,  0.0) * rcp, 0.0).rgb;
    float3 e = g_source.SampleLevel(g_sampler, uv + float2( 0.0,  1.0) * rcp, 0.0).rgb;

    // If sharpness is ~0, just return center (avoids division by zero in peak)
    if (sharp <= 0.0001)
        return float4(c, 1.0);

    // AMD CAS core: contrast-adaptive weight
    //   min/max luma around the center
    float lumaA = CasLuma(a);
    float lumaB = CasLuma(b);
    float lumaC = CasLuma(c);
    float lumaD = CasLuma(d);
    float lumaE = CasLuma(e);

    float minLuma = min(lumaC, min(min(lumaA, lumaB), min(lumaD, lumaE)));
    float maxLuma = max(lumaC, max(max(lumaA, lumaB), max(lumaD, lumaE)));

    // Contrast = max - min, clamped; adapt sharpness down in low-contrast areas
    // to avoid amplifying noise / flat shading.
    float contrast = maxLuma - minLuma;
    // AMD's peak = -1.0 / (8*D + 4*C) where D = min, C = contrast; simplified:
    // peak = -rcp(8*minLuma + 4*contrast) — more negative = stronger sharpen.
    // Scale by user sharpness (0..1). Original CAS uses sharpness 0..1 mapped
    // via ( sharp * -0.125 ) etc; we use linear scale for intuitiveness.
    float peak = -1.0 / (8.0 * minLuma + 4.0 * contrast + 0.0001);
    // Clamp peak to [-0.125, -0.0] (AMD's safe range) then scale by sharpness
    peak = clamp(peak, -0.125, -0.005) * sharp * 4.0;

    // Weighted sum: c * (1+4*peak) + (a+b+d+e)*peak  with per-channel luma-aware
    // The simple vector version is close enough for WC3's diffuse textures.
    float3 w = float3(peak, peak, peak);
    float3 res = c * (1.0 + 4.0 * peak) + (a + b + d + e) * peak;

    // Clamp to min/max to prevent halo / ringing (AMD's "soft clamp")
    float3 minRgb = min(c, min(min(a,b), min(d,e)));
    float3 maxRgb = max(c, max(max(a,b), max(d,e)));
    res = clamp(res, minRgb, maxRgb);

    // HDR guard: very bright pixels (> 4.0 linear) would blow out if sharpened;
    // lerp back toward original based on max component.
    float maxC = max(max(c.r, c.g), c.b);
    float hdrT = saturate((maxC - 4.0) * 0.5); // 4..6 linear -> 0..1
    res = lerp(res, c, hdrT);

    return float4(res, 1.0);
}
