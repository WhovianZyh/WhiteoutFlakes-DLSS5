// ============================================================================
// PN-Triangle tessellation — WhiteoutFlakes experimental
// Task 2 of WhiteoutFlakes × DLSS5 × PN-Triangle. Off by default.
//
// Pipeline: VS (skinned P/N) -> HS (uniform tess factor) -> HW Tessellator
//           -> DS (PN patch) -> PS (existing HD/SD)
// Position: bindPose -> skinning -> PN control points -> DS interpolated.
// This file provides the HS/DS pair; PS remains the existing HD/SD BLS.
// Enable via RenderSettings::PnMode (Off/x2/x4/x8/Adaptive) and
// GraphicsPipelineDesc::tessellationEnabled.
//
// Notes:
// - Uniform tessellation first (TessFactor 2/4/8), adaptive later.
// - Crease preservation via dihedral-angle threshold (pnCreaseThreshold).
// - PN patch: b030/b021/b012/b300 etc from P + N, per ATI PN-Triangles paper.
// - Shared-edge tess factor must be symmetric to avoid cracks.
// ============================================================================

import common;

struct PnVsOut {
    float3 position : POSITION;
    float3 normal   : NORMAL;
    float2 uv       : TEXCOORD0;
    float4 color    : COLOR0;
};

struct PnHsConstant {
    float edgeTess[3]    : SV_TessFactor;
    float insideTess     : SV_InsideTessFactor;
};

struct PnDsIn {
    float3 pos : POSITION;
    float3 nrm : NORMAL;
    float2 uv  : TEXCOORD0;
    float4 col : COLOR0;
};

// Cabinet for PN coefficients per patch.
struct PnPatch {
    float3 b300, b030, b003;
    float3 b210, b120, b021, b012, b102, b201;
    float3 b111;
    float3 n200, n020, n002;
    float3 n110, n011, n101;
};

cbuffer PnParams : register(b1) {
    float  pnTessFactor;      // 1,2,4,8
    float  pnCreaseDeg;       // 60.0 default: >60 preserve crease
    float  pnAdaptiveScale;   // 1.0 fixed for uniform, view-dependent later
    float  pnPad0;
};

// ---------- Utility ----------
float3 pnComputeB210(float3 p1, float3 p2, float3 n1) {
    // b210 = (2p1 + p2 - w12*n1)/3, w12 = dot(p2-p1, n1)
    float w = dot(p2 - p1, n1);
    return (2.0 * p1 + p2 - w * n1) / 3.0;
}

// ---------- Hull Shader ----------
[maxtessfactor(64.0)]
PnHsConstant pnHsConstantFunc(InputPatch<PnVsOut, 3> ip) {
    PnHsConstant o;
    // Uniform for MVP; adaptive later uses projected edge length + N·V silhouette weight.
    float f = clamp(pnTessFactor, 1.0, 8.0);
    // Hard-edge fallback: if any edge dihedral > threshold, reduce tess to avoid melting.
    // Stub: full adjacency requires extra IB/GNDX info; for now uniform.
    o.edgeTess[0] = f;
    o.edgeTess[1] = f;
    o.edgeTess[2] = f;
    o.insideTess  = f;
    return o;
}

[domain("tri")]
[partitioning("fractional_even")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("pnHsConstantFunc")]
PnDsIn pnHS(InputPatch<PnVsOut, 3> ip, uint id : SV_OutputControlPointID) {
    PnDsIn o;
    o.pos = ip[id].position;
    o.nrm = normalize(ip[id].normal);
    o.uv  = ip[id].uv;
    o.col = ip[id].color;
    return o;
}

// ---------- Domain Shader ----------
struct PnDsOut {
    float4 posH   : SV_Position;
    float3 normal : NORMAL;
    float2 uv     : TEXCOORD0;
    float4 color  : COLOR0;
};

[domain("tri")]
PnDsOut pnDS(PnHsConstant hsConst, float3 bary : SV_DomainLocation,
             const OutputPatch<PnDsIn, 3> patch) {
    // Barycentric
    float u = bary.x, v = bary.y, w = bary.z;

    // Fetch control points.
    float3 p0 = patch[0].pos, p1 = patch[1].pos, p2 = patch[2].pos;
    float3 n0 = patch[0].nrm, n1 = patch[1].nrm, n2 = patch[2].nrm;

    // Build PN patch coefficients on-the-fly (uniform path, no precompute).
    float3 b300 = p0, b030 = p1, b003 = p2;
    float3 b210 = pnComputeB210(p0, p1, n0);
    float3 b120 = pnComputeB210(p1, p0, n1);
    float3 b021 = pnComputeB210(p1, p2, n1);
    float3 b012 = pnComputeB210(p2, p1, n2);
    float3 b102 = pnComputeB210(p2, p0, n2);
    float3 b201 = pnComputeB210(p0, p2, n0);
    float3 e = (b210 + b120 + b021 + b012 + b102 + b201) / 6.0;
    float3 vv = (p0 + p1 + p2) / 3.0;
    float3 b111 = e + (e - vv) * 0.5;

    // PN position: cubic Bezier triangle.
    float3 pos =
        b300 * w*w*w + b030 * u*u*u + b003 * v*v*v +
        b210 * 3.0*w*w*u + b120 * 3.0*w*u*u + b201 * 3.0*w*w*v +
        b021 * 3.0*u*u*v + b012 * 3.0*u*v*v + b102 * 3.0*w*v*v +
        b111 * 6.0*w*u*v;

    // PN normal: quadratic interpolation (simplified).
    float3 n200 = n0, n020 = n1, n002 = n2;
    float3 n110 = normalize(n0 + n1);
    float3 n011 = normalize(n1 + n2);
    float3 n101 = normalize(n2 + n0);
    float3 nrm =
        n200 * w*w + n020 * u*u + n002 * v*v +
        n110 * 2.0*w*u + n011 * 2.0*u*v + n101 * 2.0*w*v;
    nrm = normalize(nrm);

    // UV / color linear.
    float2 uv = patch[0].uv * w + patch[1].uv * u + patch[2].uv * v;
    float4 col = patch[0].col * w + patch[1].col * u + patch[2].col * v;

    // Transform to clip (PerFrameData from common.slang; assume VS already skinned).
    // Domain shader re-projects with current view/proj.
    // Note:焊接处需在 VS 之前完成蒙皮，此处 pos 已是世界空间 skinned 位置的 PN 插值。
    // 为保持与现有管线一致，这里仅做简化的 view*proj，实际需通过 CB 传入。
    // Placeholder: passthrough to PS, real transform injected by C++ binding.

    PnDsOut o;
    o.posH   = float4(pos, 1.0); // C++ side will apply world*view*proj via CB
    o.normal = nrm;
    o.uv     = uv;
    o.color  = col;
    return o;
}
