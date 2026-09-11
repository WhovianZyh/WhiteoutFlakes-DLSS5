<p align="center">
  <img src="resources/WhiteoutFlakes.png" alt="WhiteoutFlakes" width="320">
</p>

<h1 align="center">WhiteoutFlakes</h1>

<p align="center">
  A rendering library for Warcraft III assets — classic and Reforged.
</p>

<p align="center">
  <b>Fork edition</b> — pause, parts panel, orbit camera, PN-Triangle tessellation,<br>
  CAS sharpening, DLSS5 feed and an FPS limiter, on top of upstream <code>v1.5.0</code>.
</p>

---

WhiteoutFlakes is a modular real-time renderer that reads native Warcraft III
model and texture data (`.mdx` / `.mdl`, `.blp`, `.dds`, animated sequences,
particle emitters, ribbons, splats, attachments, camera presets, day/night
cycle, …) and draws it through whichever graphics backend the platform
supports. The same library powers a standalone viewer, a 3ds Max preview
plugin, and any host that links against `WhiteoutFlakesLib`.

> **Fork notice** — this repository is a fork of
> [FernandoS27/WhiteoutFlakes](https://github.com/FernandoS27/WhiteoutFlakes),
> rebased onto upstream tag [`v1.5.0`](https://github.com/FernandoS27/WhiteoutFlakes/releases/tag/v1.5.0)
> (`02abea8`). Development happens on the default branch
> `feature/dlss5-pn-triangle` (see [Branches](#branches)). Everything upstream
> ships with is intact — this fork only adds.

## What this fork adds

| Feature | Notes |
| --- | --- |
| **Pause** | `Space` freezes the whole scene; UI localized in 11 languages. |
| **Parts panel** | Per-geoset visibility toggles (View → Parts Panel) — hide armor, weapons or any block at runtime, no model edits needed. |
| **Bloom / FXAA / SMAA** | NVIDIA FXAA 3.11 and SMAA (iryoku 2013), applied pre-tonemap on HDR color; FXAA and SMAA are mutually exclusive, Bloom stacks on top. |
| **CAS sharpening** | AMD FidelityFX CAS v1.3 (MIT), HDR path — runs after bloom/AA and before tonemap; toggle + 0–1 sharpness, persisted. |
| **Orbit camera** | Manual / Slow / Medium / Fast (0.12 / 0.30 / 0.65 rad/s), camera-orbit or model-rotate, controlled from the toolbar and settings window, persisted in `WhiteoutFlakes.ini`. |
| **Orbit capture export** | One menu item renders a 360° turntable — 120 frames @ 30 fps after a 10 s warmup — straight to a folder you pick. |
| **PN-Triangle tessellation** | Off / ×2 / ×4 / ×8 / **Adaptive** (600/1200/2400 world-unit distance tiers); D3D11 hull/domain shaders compiled at runtime — view-space HS input feeding a world-space Bezier domain shader, so clipping and silhouettes stay correct while curvature gets smoothed. |
| **DLSS5 neural rendering feed** *(experimental, D3D11)* | ReShade 6.8 depth + LumeniteFX motion vectors → DLSS5 feed → NVIDIA DLAA. The runtime components (ReShade `dxgi.dll`, `dlss5-feed.addon64`, `nvngx_dlss*.dll`) are **not redistributed** in this repo; enable via `NeuralRenderingEnabled` in `WhiteoutFlakes.ini`. |
| **FPS limiter** | Unlimited / 30 / 60 / 120 / 144 / 240 (default 120), enforced in the main loop. |

### Hotkeys

Applied on the next frame, no restart, persisted to the INI.

| Key | Action |
| --- | --- |
| `Space` | Pause / resume |
| `F1` | Cycle PN mode: Off → ×2 → ×4 → ×8 → Adaptive |
| `F2` | Cycle anisotropic filtering: 1 / 2 / 4 / 8 / 16 |
| `F3` | Toggle CAS sharpening |
| `F4` | Cycle AA: Off → FXAA → SMAA → Off |
| `F5` | Toggle Bloom |
| `F6` | Cycle mip bias: −0.6 / −0.3 / 0.0 |
| hold `Tab` | A/B compare — original image (AF / CAS / PN / AA off) |

## Screenshots

<p align="center">
  <img src="resources/screenshots/animated_01.webp" alt="WhiteoutFlakes preview" width="560">
</p>

<details>
<summary><b>More screenshots</b></summary>

<table>
  <tr>
    <td><img src="resources/screenshots/animated_02.webp" alt="Screenshot 2" width="380"></td>
    <td><img src="resources/screenshots/animated_03.webp" alt="Screenshot 3" width="380"></td>
  </tr>
  <tr>
    <td><img src="resources/screenshots/animated_04.webp" alt="Screenshot 4" width="380"></td>
    <td><img src="resources/screenshots/animated_05.webp" alt="Screenshot 5" width="380"></td>
  </tr>
  <tr>
    <td><img src="resources/screenshots/animated_06.webp" alt="Screenshot 6" width="380"></td>
  </tr>
</table>

</details>

## What it renders

- **MDX / MDL models** — classic (v800) and Reforged HD (v900, v1000, v1100, v1200),
  including multiple texture slots, fresnel terms, emissive gain, layer
  flipbooks, and per-vertex tangent frames.
- **Skeletal animation** — Hermite / Bezier / Linear tracks, global
  sequences, bone constraints, IK-free; multiple actors with independent
  timelines.
- **Particle emitters** — PartcileEmitter1, ParticleEmitter2, CornEffects
  Reforged effects via an interpolarity layer called cornflakes
  simulation runtime; ribbons, splats, projected decals, billboards.
- **Replaceable textures** — team color, team glow, tilesets (16 regions),
  cliff sets, water; the live-reload material path the Max plugin exposes
  re-skins models without restart.
- **Day / night cycle + IBL** — Portrait / Day-Night / Dungeon / Sunset
  probe sets, shadow cascades (0–3), tonemap, three lighting modes
  (InGame / Glue / Dynamic).
- **Camera presets** — scripted MDX cameras with optional animators, plus a
  ViewCube widget for free-orbit navigation.

## Graphics backends

| Backend | Platform | Notes |
| --- | --- | --- |
| D3D12   | Windows | Default on Windows. |
| D3D11   | Windows | Fallback for older drivers; the PN-Triangle and DLSS5 paths in this fork target D3D11. |
| Vulkan  | Windows / Linux / macOS | macOS via MoltenVK; primary backend on Linux. |
| Metal   | macOS   | Native backend — default on macOS. |
| WebGPU  | Browser | Emscripten + emdawnwebgpu; powers the web viewer. |

The renderer abstracts every backend behind a unified `gfx::IGFXDevice`
interface; the engine itself never sees an `HWND` / `VkDevice` / `ID3D12*` /
`MTLDevice` / `WGPUDevice`. Shaders are compiled once from Slang sources
into BLS bundles that target DXBC / DXIL / SPIR-V / MSL / WGSL in parallel;
the prebuilt pack ships under [`prebuilt/shaders/`](prebuilt/shaders) so a
fresh clone can render without installing the Slang toolchain.

## Hosts

- **[`tools/basic_viewer/`](tools/basic_viewer/) `WhiteoutFlakes` standalone** — GLFW window + Dear ImGui UI, file picker via
  `nativefiledialog-extended`, cubeb-backed audio with 3D sound. Cross-platform.
- **[`tools/max_plugin/`](tools/max_plugin/) `WhiteoutFlakes.dlx` 3ds Max plugin** — Win32 host with the same Dear
  ImGui surface; lives next to the modeler, hot-reloads materials.
- **[`tools/web_viewer/`](tools/web_viewer/) browser viewer** — Emscripten / WebGPU build (`wf-core.{js,wasm}`)
  driven by a small ES module facade that mirrors mdx-m3-viewer's shape.
  Assets stream from a local picked directory and/or Hiveworkshop's CASC
  mirror; Web Audio handles SND events. The matching
  [`tools/web_viewer/casc_server/`](tools/web_viewer/casc_server/) is a
  Crow-based dev server that serves loose files out of a local WC3 install
  for offline iteration.

## Building

### Quick start (Windows / MSVC)

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target WhiteoutFlakesStandalone
```

The standalone viewer lands at `build/standalone/Release/WhiteoutFlakes.exe`.

Notes from this fork's Windows development (VS 2022 Build Tools):

- Compile with `/utf-8` — sources and build paths carry non-ASCII text.
- `NOMINMAX` must be defined before including `d3dcompiler.h`, or macro
  pollution of `std::max` breaks `render_pipeline.cpp`.
- The development configuration targets D3D11/D3D12 and keeps the Vulkan
  backend disabled, but header-only Vulkan-Headers must still be reachable
  for the build.

### Web viewer (Emscripten / WebGPU)

```
emcmake cmake -S . -B build-web -G Ninja
cmake --build build-web --target wf_web
python tools/web_viewer/serve_nocache.py 8080   # then open http://localhost:8080
```

`wf_web` outputs `build-web/web/wf-core.{js,wasm}` plus the staged JS facade
and `index.html`. Asset delivery defaults to Hive's CASC mirror; for offline
work, build `wf_casc_server` (`-DWDX_BUILD_CASC_SERVER=ON`) and point the
viewer at it.

### Other toolchains

| Toolchain | Tested |
| --- | --- |
| MSVC 2022 | ✓ (primary) |
| Clang 21 (LLVM) + Ninja | ✓ |
| MinGW UCRT64 + Ninja | ✓ |
| GCC 15 (Linux) + Ninja | ✓ via CI |
| AppleClang 15 (macOS 13.3+) | ✓ first-class — Metal backend, native arm64 build, signed `.dmg` |
| Emscripten 4.0.10+ + emdawnwebgpu | ✓ web viewer build (`-DEMSCRIPTEN=ON`) |

### Useful CMake options

| Option | Default | Purpose |
| --- | --- | --- |
| `WDX_BUILD_WC3_SHADERS`        | `OFF` | Run slangc and rebuild the BLS bundles from `externals/Wc3Shaders/`. |
| `WDX_USE_PREBUILT_SHADERS`     | auto  | Use the committed `prebuilt/shaders/` pack (auto-enabled when the dir exists and shaders aren't being built from source). |
| `WDX_BUILD_WC3_DEBUG_SHADERS`  | `ON`  | Also stage debug-symbol BLS bundles for the renderer's graphics-debug mode. |
| `WDX_ENABLE_TRACY`             | `ON`  | Link Tracy profiler client (`TRACY_ENABLE`, `TRACY_ON_DEMAND`). |
| `WDX_ENABLE_IMGUI`             | `ON`  | Engine-side BLS-backed Dear ImGui adapter + GLFW/Win32 frontends. |
| `WDX_BUILD_MAX_PLUGIN`         | `OFF` | Build the 3ds Max plugin (Windows only; needs `-DMAX_VERSION=<year>`). |
| `WDX_BUILD_CASC_SERVER`        | `OFF` | Build `wf_casc_server` — local dev replacement for Hive's CASC delivery. |

## Branches

| Branch | Contents |
| --- | --- |
| `feature/dlss5-pn-triangle` *(default)* | Active development — the baseline below plus PN-Triangle tessellation, the DLSS5 feed, orbit camera + capture, CAS, the FPS limiter and real-time hotkeys. |
| `main` | Baseline: upstream `v1.5.0` (`02abea8`) + vendored externals + the first wave of viewer customizations (pause / parts panel / bloom / FXAA / SMAA). |

Unlike upstream — which fetches `externals/` as submodules — the submodule
contents are committed in-tree in this fork, so a plain `git clone` (no
`--recursive`) is enough to build.

## Packaging

Prebuilt artifacts are produced by [GitHub Actions](.github/workflows/):

- **`linux-appimage.yml`** — Ubuntu 24.04 + GCC 15 + LunarG SDK 1.4.341.0;
  output: `WhiteoutFlakes-linux-x86_64.AppImage`.
- **`macos-dmg.yml`** — macOS 14 (Apple Silicon) + AppleClang. Ships the
  native Metal backend by default; Vulkan-via-MoltenVK is also linked in
  via LunarG SDK 1.4.341.0 for backend-bring-up comparison. Output:
  `WhiteoutFlakes-macos-arm64.dmg`, drag-and-drop installer with the .app,
  ad-hoc signed.

## Project layout

```
src/
  gfx/          Backend-agnostic graphics interface; D3D11 / D3D12 /
                Vulkan / Metal / WebGPU implementations.
  renderer/     Engine: pipeline, scene, BLS shader cache, particle system,
                shadow + IBL services, cornflakes (Reforged effects runtime).
  io/           MDX parsing adapter, BLP/DDS/TGA loaders, CASC/MPQ provider.
  public_api/   Stable C++ ABI used by external hosts (ActorView, etc.).

tools/
  basic_viewer/ Standalone GLFW + ImGui viewer.
  max_plugin/   3ds Max .dlx plugin.
  web_viewer/   Emscripten/WebGPU browser host + JS facade + service
                worker; the casc_server/ subdir is a Crow-based local
                stand-in for Hive's CASC delivery.
  common/       Shared host utilities (cubeb sound emitter, ImGui theme).

externals/      Submodules: WhiteoutLib (MDX/CASC/MPQ), Wc3Shaders, GLFW,
                Dear ImGui, cubeb, Tracy, nativefiledialog-extended.

prebuilt/       Pre-compiled BLS shader pack + warmed-up PSO trace, so
                CI / fresh clones don't need the Slang toolchain.

packaging/      Linux .desktop + macOS Info.plist template.
```

## Status

Active development on the default branch. The renderer is feature-complete
for classic and Reforged MDX content; the fork's rendering experiments
(PN-Triangle, DLSS5) target D3D11.

## License

See [`LICENSE`](LICENSE) for project terms and
[`LICENSE-AI.md`](LICENSE-AI.md) for the AI-tooling disclosure.
WhiteoutFlakes bundles a number of third-party libraries under their own
licenses; consult each submodule under [`externals/`](externals/) for
details. This fork carries the same BSD-3-Clause license; all fork-specific
changes are likewise distributed under BSD-3-Clause.

> *Warcraft III is a trademark of Blizzard Entertainment, Inc.
> WhiteoutFlakes is an independent project not affiliated with or endorsed
> by Blizzard. The renderer reads only assets the user already owns.*
