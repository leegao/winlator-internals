# Winlator Internals

I'm a bored (former) software engineer who loves to take things apart. In this series, I'm going to reverse engineer Winlator and its various forks, libraries, and dependencies.

See https://leegao.github.io/winlator-internals/

---

## Series

1. [Vortek Internals: Part 1 - Command Buffers](https://leegao.github.io/winlator-internals/2025/06/01/Vortek1.html)
   * Deep dive into the internal architecture of Vortek, a Vulkan "driver" designed to work around runtime incompatibilities of software running within glibc on Bionic systems within Winlator. Vortek implements a client-server model where Vulkan commands are marshaled across an IPC boundary, allowing a game client running on glibc (box64 + wine) to interface with a native Vulkan renderer server (winlator and surfaceflinger).
2. [Vortek Internals: Part 2 - Driver-Specific Workarounds](https://leegao.github.io/winlator-internals/2025/06/02/Vortek2.html)
   * Deep dive into some driver-specific workarounds that Vortek uses to enable DirectX gaming on non-spec-compliant hardware, such as Mali GPUs, by emulating missing features directly within the driver, addressing shortcomings like the absence of BCn texture compression, gl_ClipDistance, etc.
