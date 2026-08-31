// ============================================================================
// Velocity / Motion Vector generation — WhiteoutFlakes experimental
// Task 3 of WhiteoutFlakes × DLSS5 × PN-Triangle.
//
// Goal: produce per-pixel velocity in RG16F for DLSS5-Feeder / ReShade.
// Two modes:
//   OpticalFlow (default) — external iMMERSE LaunchPad estimated MV.
//   RendererTrueMV — current vs previous clip-space (camera+skeletal).
//
// Pipeline spot: MRT slot or standalone pass after main opaque geoms, before
// post-process. Format: R16G16_FLOAT (pixel delta), NDC -> UV conversion on
// write. Must match Feeder convention; flip/scale exposed for debug.
//
// History required (RenderPipeline::Impl):
//   prevView, prevProj, prevModel, prevBonePalette[~256], prevBillboardBasis
// Skeletal: p(t) = Σ wi*Bi(t)*p(bind), p(t-1) = Σ wi*Bi(t-1)*p(bind)
// PN consistency: if PN enabled, evaluate PN patch for BOTH current and
// previous skinned positions in DS and output currClip/prevClip together.
// ============================================================================

import common;

cbuffer VelocityParams : register(b2) {
    float4x4 prevViewProj;     // PrevProj * PrevView * PrevModel (or per-bone)
    float4x4 currViewProj;     // CurrProj * CurrView * CurrModel
    float2   velocityScale;    // 0.5 for NDC->UV, 1.0 for pixel
    float2   velocityFlip;     // (flipX, flipY) : +1 or -1
    float    velocityPad0;
    float    velocityPad1;
};

struct VelocityVsIn {
    float3 pos     : POSITION;
    float3 normal  : NORMAL;
    float2 uv      : TEXCOORD0;
    uint4  boneIdx : BLENDINDICES0;
    float4 boneW   : BLENDWEIGHT0;
};

struct VelocityVsOut {
    float4 currPosH : SV_Position;
    float4 prevPosH : TEXCOORD0; // prev clip for PS velocity calc
    float2 uv       : TEXCOORD1;
};

// CPU supplies prevBonePalette as structured buffer for skeletal.
// Stub: single prevViewProj for rigid/camera paths; skeletal path samples
// palette in DS/PS per vertex.
VelocityVsOut velocityVS(VelocityVsIn vin) {
    VelocityVsOut o;

    // Rigid/camera path: p is already skinned on CPU or via palette in VS.
    float4 p = float4(vin.pos, 1.0);
    o.currPosH = mul(currViewProj, p);
    o.prevPosH = mul(prevViewProj, p);
    o.uv       = vin.uv;
    return o;
}

struct VelocityPsOut {
    float2 velocity : SV_Target0; // RG16F
};

// NDC = clip.xy / clip.w, velocity = (currNDC - prevNDC) * scale * flip
float2 velocityFromClip(float4 currClip, float4 prevClip, float2 scale, float2 flip) {
    float2 currNdc = currClip.xy / max(currClip.w, 1e-6);
    float2 prevNdc = prevClip.xy / max(prevClip.w, 1e-6);
    float2 v = (currNdc - prevNdc) * scale;
    v *= flip;
    // For pixel-space convention, multiply by viewport size elsewhere.
    return v;
}

VelocityPsOut velocityPS(VelocityVsOut pin) {
    VelocityPsOut o;
    // UV-space velocity (default): scale = 0.5 because NDC [-1,1] -> UV [0,1] is 0.5x
    // Feeder may expect pixel-space: scale = viewportDim * 0.5
    o.velocity = velocityFromClip(pin.currPosH, pin.prevPosH, velocityScale, velocityFlip);
    return o;
}

// Domain shader variant for PN consistency (see pn_triangle.slang):
// When PN is enabled, evaluate PN for BOTH curr and prev skinned triangles
// in DS and emit currPosH/prevPosH together, so velocity matches final
// tessellated silhouette.
