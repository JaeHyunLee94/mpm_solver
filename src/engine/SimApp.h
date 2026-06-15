//
// SimApp.h – thin helpers that abstract away backend (CUDA vs OpenMP) and
// display (OpenGL vs headless) differences so scene files stay readable.
//
// Usage in a scene file:
//
//   #include <SimApp.h>
//
//   // In run():
//   mpm::engineStep(*engine, dt);          // picks CUDA or CPU automatically
//
//   // Headless main loop:
//   mpm::headlessRun(*engine, dt, 10000);  // run for N frames, print progress
//
#pragma once

#include "Engine.h"
#include <fmt/core.h>

namespace mpm {

// ---------------------------------------------------------------------------
// engineStep – calls integrateWithCuda when CUDA is compiled in, otherwise
// falls back to the OpenMP CPU path.
// ---------------------------------------------------------------------------
inline void engineStep(Engine& engine, float dt) {
#ifdef MPM_CUDA_AVAILABLE
    engine.integrateWithCuda(dt);
#else
    engine.integrate(dt);
#endif
}

// ---------------------------------------------------------------------------
// headlessRun – a simple loop for HPC / no-display environments.
//   max_frames : stop after this many frames (0 = run forever)
//   print_every: print a progress line every N frames
// ---------------------------------------------------------------------------
inline void headlessRun(Engine& engine, float dt,
                        unsigned long long max_frames = 0,
                        unsigned long long print_every = 500) {
    engine.resume();  // engine starts paused; must call resume() to begin
    fmt::print("[MPM headless] starting simulation  dt={:.2e}  max_frames={}\n",
               dt, max_frames);

    while (true) {
        engineStep(engine, dt);
        const auto frame = engine.getCurrentFrame();
        if (print_every > 0 && frame % print_every == 0)
            fmt::print("[MPM headless] frame {}\n", frame);
        if (max_frames > 0 && frame >= max_frames)
            break;
    }

    fmt::print("[MPM headless] finished at frame {}\n", engine.getCurrentFrame());
}

} // namespace mpm
