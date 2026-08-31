# WhiteoutFlakes × DLSS5 × PN-Triangle 工程规划（基于真实代码）

> 基于: `/mnt/c/Users/zyh/Downloads/WhiteoutFlakes_DLSS5_PN_Triangle_实施方案.md` (2026-08-31)
> 代码基线: `FernandoS27/WhiteoutFlakes` v1.5.0 + 本地暂停/部件面板/FXAA/SMAA 定制 (branch `feature/dlss5-pn-triangle` @ `e1db629`)
> 平台: Windows x64 / RTX 5070 Ti Laptop 12GB / VS2022 / Slang

---

## 0. 方案与真实代码的对齐

实施方案第3章假设 `renderer_test/d3d11/RendererD3D11.cpp` 等路径，**实际仓库已重构**：

| 方案假设 | 真实仓库位置 | 说明 |
|---|---|---|
| `renderer_test/d3d11/Main.cpp` | `tools/basic_viewer/test_main.cpp` + `tools/basic_viewer/viewer_app.cpp` | Host 入口，GLFW+ImGui，调用 `RenderPipeline::InitDevice(api)` |
| `renderer_test/d3d11/RendererD3D11` | `src/gfx/d3d11/{d3d11_device, d3d11_command_list}` + `src/renderer/render_pipeline.{h,cpp}` | 设备抽象 `gfx::IGFXDevice`，D3D11 仅为 fallback，后端选择存于 `RenderSettings::DefaultBackend()` + `WhiteoutFlakes.ini: DefaultBackend` |
| `mdx_test/src/camera.cpp` | `src/renderer/camera.h` + `tools/basic_viewer/viewer_app.cpp:FrameCameraToModel()` | 单头文件 Camera，无独立 cpp |
| Mesh shader / batching | `src/renderer/bls/*`, `src/renderer/render_detail.cpp:BuildDrawLists()`, `src/renderer/model/model_instance.h` | HD 经过 BLS，SD 直接管线，hiddenGeosets 已在此层 |
| Shaders `basic_mesh.hlsl` | `src/renderer/shaders/*.slang` (blit, fxaa, smaa, gtao, gaussian_blur, line) + `externals/Wc3Shaders` + `prebuilt/shaders/` BLS 包 | Slang 编译为 BLS，`compiled_shaders.h` |

**结论**：MVP 仍用 D3D11，但真实切入点是 `gfx::GfxApi::D3D11` + `RenderPipeline` + `Camera` + `RenderSettings` + `viewer_ui`，而非独立的 renderer_test。

---

## 1. 总架构（修正后）

```
MDX (不改)
  ↓ WhiteoutLib
ViewerApp (GLFW) → RenderService → RenderPipeline → gfx::IGFXDevice (D3D11/D3D12)
  ↓
Skinning (CPU, animation/actor_eval)
  ↓
[Task2] PN-Triangle + Tessellation (HS/DS, 可选)
  ↓
Raster → hdrColor(R11G11B10) + depth + (Task3: velocity RG16F)
  ↓
PostProcess: Bloom → FXAA/SMAA (已在 hdrColor, pre-tonemap)
  ↓
Tonemap → SwapChain
  ↓
[Task1] ReShade → Depth + [Task3 前用 OpticalFlow / Task3 后用 True MV] → DLSS5-Feeder → renodx-dlss5 addon → Neural
```

与方案一致，但强调 **D3D11 路径需显式验证**：`WhiteoutFlakes.ini: DefaultBackend=d3d11` 或 `--backend d3d11`。

---

## 2. 四大任务在真实代码中的落点

### 任务一：D3D11 + DLSS5 基线（不改核心几何）

| 子任务 | 真实文件 | 动作 |
|---|---|---|
| 1.1 梳理管线 | `src/gfx/gfx.h`, `src/renderer/render_pipeline.cpp`, `tools/basic_viewer/viewer_app.cpp:Open()` | 确认 `InitDevice(D3D11)` 创建 device/swapchain/depth/colorLinear/hdrColor，追踪 `RenderViewport()` 全链 |
| 1.2 固定基线场景 | `viewer_app.cpp:FrameCameraToModel()`, `render_settings.h:LightingMode` | 选一个不透明 humanoid（如 `Hero Naga Royal Guard Reforged/Blue_Yellow`），固定 `LightingMode=Glue:1`, 背景, 1080p/1440p |
| 1.3 自动 Orbit Camera | `src/renderer/camera.h` + 新增 `src/renderer/orbit_camera.*` + `tools/basic_viewer/viewer_ui.cpp` + `render_settings.h` | 见下节详述 |
| 1.4 ReShade 验证 | 外部：安装 ReShade 6.x, 勾选 `Generic Depth`, 确认 scene depth | 若 depth 错误，查 `RenderTarget` depthFormat (R24→R32 fallback, AMD 特殊) |
| 1.5 DLSS5-Feeder | 外部：ReShade → iMMERSE LaunchPad (estimated MV) → DLSS5-Feeder (x64 D3D11) → renodx-dlss5 addon | Feeder 需 `color+depth+estimatedMV`，第一版不需自制 MV/G-buffer |

**成功判据**：同轨迹 `Baseline OFF vs DLSS5 ON`，1440p 慢速 Orbit 肉眼可见 skin/cloth/armor 细节增强（允许轻度 ghosting）。

### 任务二：PN-Triangle（真实轮廓）

* **着色器**：新增 `src/renderer/shaders/pn_triangle.slang`（HS/DS），或扩展现有 `hd` BLS 管线分支。Uniform TessFactor 2/4/8 起步，后续视距自适应。
* **GFX 层**：扩展 `src/gfx/gfx_pipeline_types.h:GraphicsPipelineDesc` 增加 `HullShader/DomainShader` 与 `PrimitiveTopologyType::Patch`，各后端 `CreateGraphicsPipeline` 适配（先 D3D11/D3D12，Vulkan/Metal 后补）。
* **BLS 层**：`src/renderer/bls/bls_pso_builder.cpp` 增加 tessellation PSO 变体，`bls_permuter` 增加 `PnMode` 排列。
* **CPU 侧**：`render_detail.cpp:BuildDrawLists()` 后按 geoset 分类（opaque/alpha-test 可做，particle/ribbon/additive 首次跳过），`hiddenGeosets` 同层过滤。
* **顺序**：`BindPose → Skinning → PN patch → Tessellation → Raster`（方案 3.5 节），不做 `BindPose PN → Skin`。
* **Hard Edge**：`geoset_classify.cpp` 增加材质/法线 discontinuity 阈值 `<30° smooth / 30-60° reduced / >60° crease`，按 lodName/材质边界调权重，UI 可调。
* **UI**：`RenderSettings` 新增 `PnMode {Off,x2,x4,x8,Adaptive}` + `viewer_ui.cpp` 下拉，`settings_ini.cpp` 持久化。

四组对照：`Original / PN / DLSS5 / PN+DLSS5`，以头/肩/臂/腿/有机曲面是否更圆滑为判据。

### 任务三：Renderer True Motion Vectors

* **资源**：新增 `Velocity Texture (RG16F)`，分辨率同 `hdrColor`，存于 `RenderTarget` 或 `RenderPipeline::Impl`。
* **历史状态**：`Previous View/Proj`, `Previous Model`, `Previous Bone Palette` ( `animation/actor_eval.cpp` 骨骼表 ), `Previous Billboard Basis`。存于 `RenderPipeline::Impl` 或新增 `VelocityService`。
* **计算**：
  ```hlsl
  currClip = CurrProj*CurrView*CurrModel*currPos;
  prevClip = PrevProj*PrevView*PrevModel*prevPos;
  motion = currNdc - prevNdc; // 再转 UV/Pixel，暴露 X/Y flip/scale debug
  ```
  Skeletal：`p(t)=Σ wi*Bi(t)*p(bind)` vs `p(t-1)`，在 VS/DS 内双次 skinning。
* **PN 一致性**：若开 PN，VS→HS→DS 后再算 motion，DS 同时输出 `CurrentClip/PreviousClip`，写入 Velocity 的是 **最终 tessellated 曲面** 的运动，否则 `Color silhouette ≠ Velocity silhouette` 会加重 ghosting。
* **接给 Feeder**：Viewer 增加 `MV Source [OpticalFlow / Renderer True MV]` 切换，需微调 Feeder/ReShade effect 适配外部 velocity 输入；初期可在 ReShade 侧用外部纹理替换。
* **暂不完美**：Billboard 需存 Facing Basis，Particle/Ribbon 首次可 `no MV` 或后续加 reactive mask。

成功判据：`Estimated MV vs True MV` 在 Idle/Walk/Attack + Camera Orbit/Model Rotate 下，ghosting/texture swimming/detail popping 明显收敛。

### 任务四：统一测试与是否进 D3D12

* **必测矩阵**：1-4（PN×DLSS5）+ 5-8（MV 源切换），每模型跑 `Slow/Medium/Fast × CameraRotate/ModelRotate × Idle/Walk/Attack/Spell/Death`。
* **模型集**：humanoid, creature, armor-heavy, building, foliage, particle-heavy（前三类决定价值）。
* **记录**：`build,asset,animation,camera_mode,pn_mode,tess_factor,mv_mode,resolution,nr_enabled,fps,gpu_ms,vram_mb,gpu_util,nr_status` CSV（方案 §4.3）。
* **进 D3D12 门槛**：DLSS5 有收益 + PN 有改善 + True MV 有改善 + 1440p 可交互，才迁移 `Native D3D12 Color+Depth+Velocity → Native NGX`。

---

## 3. 第一阶段可交付的最小实现切片（本次分支）

为让 `feature/dlss5-pn-triangle` 立即可编译、可演示，按依赖排序：

1. **Orbit Camera（Task 1.3）** — 零侵入纯新增，最易验证：
   - 新增 `src/renderer/orbit_camera.h/.cpp`（环绕控制器，封装 `Camera` 的 yaw/pitch/distance 驱动）
   - `RenderSettings` 加 `OrbitMode {Manual, Slow, Medium, Fast}` + `OrbitAxisMode {CameraOrbit, ModelRotate}`
   - `viewer_app.cpp:Tick()` 每帧按 `dt * speed` 递增 `azimuth 0→360°`，`elevation/radius/target` 固定，复用现有 `Camera::SetYaw/SetPitch/SetTarget`
   - `viewer_ui.cpp` 增加 `Camera/Orbit` 下拉，`settings_ini.cpp` 持久化，INI 键 `OrbitMode, OrbitSpeed, OrbitAxisMode`
   - 复用现有 `DefaultBackend` 切换逻辑，可 A/B 同轨迹

2. **PN-Triangle 桩（Task 2 骨架）** — 先搭管线与开关，不急于算法完美：
   - `gfx_pipeline_types.h` 加 tessellation 字段（HS/DS handle, PatchControlPoints=3）
   - 新增 `shaders/pn_triangle.slang` 最简 PN 插值（position + normal 二次 Bezier），HS 固定 `TessFactor` 2/4/8，DS 输出
   - `bls_pso_builder` 加一个 `PnEnabled` 分支，`render_settings.h` 加 `PnMode`
   - `viewer_ui` 加 `Geometry [Off/PN x2/x4/x8]`，默认 Off，回退语义保证关闭时渲染路径不变

3. **Velocity Texture 桩（Task 3 骨架）** — 先分配资源与 Previous 矩阵，不急于 skeletal 双 skinning：
   - `render_target`/`render_pipeline_impl` 加 `velocity` 纹理（`R16G16_FLOAT`）、`prevView/prevProj`
   - 每帧 `RenderViewport` 开头保存 `prev = curr`，`post_process` 阶段若开 True MV 则写 velocity（首版可仅 camera/rigid 运动，后补 skeletal）
   - `RenderSettings` 加 `MvSource {OpticalFlow, TrueMV}` + debug 可视化开关

> 注意：DLSS5 runtime / Feeder / ReShade 均为外部闭源/社区分发，本仓库 **不提交 `nvngx_dlssnr.dll` 或 WC3 资产**，仅提供对接说明文档与外部纹理共享约定。

---

## 4. 风险与缓解（对应方案第8章）

* **R1 Feeder/NGX 版本飘**：`bls` 与 `frame_capture` 已通过外部纹理隔离，DLSS5 始终 `Optional Experimental`，关闭后管线不变。
* **R2 风格漂移**：评测表同时记 `Modernization Score` vs `Style Preservation Score`，不单看清晰度。
* **R3 PN 过融**：默认 Off，按 geoset/material 分类，武器/机械件阈值调高，`hiddenGeosets` 可单块关闭验证。
* **R4 PN↔MV 不一致**：规范 `Color 与 Velocity 均取最终 tessellated 位置`，首版若未合入需在文档标注 `Velocity 仍为 coarse` 的已知限制。

---

## 5. 执行顺序（本分支）

```
1. Orbit Camera  ← 本次先做，可独立编译验证
2. PN 桩 + 开关
3. Velocity 桩 + Previous 矩阵
4. 统一 A/B 与 CSV 记录
5. ReShade + Feeder 外部联调（人工）
6. 通过前5项后再评估 Native D3D12
```

---

## 6. 验证方式

* 编译：`cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ... && cmake --build build --config Release --target WhiteoutFlakesStandalone`
* 运行：`--backend d3d11` 或 INI `DefaultBackend=d3d11`，打开任意 MDX，切换 `Camera: Orbit Slow/Medium/Fast` 观察 360° 轨迹是否稳定；`--res 1920×1080 / 2560×1440` 对比；`ReadbackTarget` / `FrameCapture` 出图做 A/B。
* 回退：所有新增开关默认 Off/Fast 路径关闭时，渲染结果与 `main` 逐像素一致（除 UI 新增控件）。

---

## 7. 文件清单（预期改动）

* 新增：`src/renderer/orbit_camera.h`, `src/renderer/orbit_camera.cpp`, `src/renderer/shaders/pn_triangle.slang`
* 修改：`src/renderer/camera.h`（暴露 Orbit 辅助）、`src/renderer/render_settings.h`、`.cpp`、 `src/gfx/gfx_pipeline_types.h`、 `src/gfx/d3d11/d3d11_device.*`、`src/renderer/render_target.*`、`src/renderer/render_pipeline.*`、`src/renderer/bls/*`、`tools/basic_viewer/viewer_app.{h,cpp}`、`tools/basic_viewer/viewer_ui.cpp`、`tools/basic_viewer/settings_ini.{h,cpp}`、`CMakeLists.txt`（可选 shader entry）
* 不改：`externals/WhiteoutLib`、`externals/Wc3Shaders`、`prebuilt/`（除非重编 BLS）

---

## 8. 下一步（给 Coding Agent）

按本规划 §3 顺序逐项提交 PR，每项保证 **单次编译通过**。人工负责 ReShade 深度选对、Feeder/NGX 版本、裂缝/蒙皮异常的 RenderDoc 定性。不要并行铺开 Vulkan/Metal/WebGPU/PBR/离线重建。
