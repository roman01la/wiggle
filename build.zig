const std = @import("std");

pub fn build(b: *std.Build) void {
    // Target WebAssembly for the browser (freestanding)
    const target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
    });

    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "wiggle",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // Export all functions marked with `export`
    exe.entry = .disabled;
    exe.rdynamic = true;

    // Install the wasm file to zig-out/bin/
    b.installArtifact(exe);

    // Add a custom step to copy to www directory
    const copy_wasm = b.addInstallFile(exe.getEmittedBin(), "../www/wiggle.wasm");

    const install_step = b.step("www", "Build and copy WASM to www directory");
    install_step.dependOn(&copy_wasm.step);

    // ========================================================================
    // Convenience steps for debug and release builds
    // ========================================================================

    // Debug build step (fast compile, debug symbols, no optimization)
    const debug_exe = b.addExecutable(.{
        .name = "wiggle",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = .Debug,
        }),
    });
    debug_exe.entry = .disabled;
    debug_exe.rdynamic = true;
    const copy_debug_wasm = b.addInstallFile(debug_exe.getEmittedBin(), "../www/wiggle.wasm");
    const debug_step = b.step("debug", "Fast debug build (unoptimized, fast compile)");
    debug_step.dependOn(&copy_debug_wasm.step);

    // Release build step (optimized for speed)
    const release_exe = b.addExecutable(.{
        .name = "wiggle",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        }),
    });
    release_exe.entry = .disabled;
    release_exe.rdynamic = true;
    const copy_release_wasm = b.addInstallFile(release_exe.getEmittedBin(), "../www/wiggle.wasm");
    const release_step = b.step("release", "Optimized release build (fast execution)");
    release_step.dependOn(&copy_release_wasm.step);

    // Small release build (optimized for size)
    const small_exe = b.addExecutable(.{
        .name = "wiggle",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = .ReleaseSmall,
        }),
    });
    small_exe.entry = .disabled;
    small_exe.rdynamic = true;
    const copy_small_wasm = b.addInstallFile(small_exe.getEmittedBin(), "../www/wiggle.wasm");
    const small_step = b.step("small", "Size-optimized release build (smallest file)");
    small_step.dependOn(&copy_small_wasm.step);
}
