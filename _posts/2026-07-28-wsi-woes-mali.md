---
title: "WSI Woes on Mali"
date: 2026-07-28
categories: [wrapper]
tags: [mali, vulkan, wsi, winlator, android, bionic-wrapper]
---

This document contains the investigations for [Fuel and other 32-bit games don't work · Issue #157 · leegao/bionic-vulkan-wrapper](https://github.com/leegao/bionic-vulkan-wrapper/issues/157) as well as an overview of the WSI implementation in major Winlator forks.

**PRs**:
* [#5 - Fix ion heap UAPI](https://github.com/GameNative/mesa/pull/5)
* [#6 - Fix swapchain blit and DR fallback](https://github.com/GameNative/mesa/pull/6)
* [#7 - Fix multi-handle AHB mmap](https://github.com/GameNative/mesa/pull/7)
* [#8 - Unlink dedicated allocation structs](https://github.com/GameNative/mesa/pull/8)


## So What’s the Big Deal?



The [Winlator Bionic wrapper](https://github.com/Pipetto-crypto/mesa/blob/wrapper-25/src/vulkan/wrapper/), and its [precursor](https://github.com/xMeM/mesa/tree/wrapper), both suffer from a couple of problems on Mali devices:

* On phones before GKI [android12-5.10](https://source.android.com/docs/core/architecture/kernel/gki-android12-5_10-release-builds) devices (basically any device produced before 2021), 32-bit games will either crash at `vkCreateSwapchainKHR`, or else black screen during game launch.


* On all Mali devices, the wrapper WSI implementation (the [Mesa Vulkan WSI runtime](https://github.com/leegao/mesa-wrapper-CI/blob/05bca95965ccebfeb2d34e415f5c5919e790a875/src/vulkan/wsi/wsi_common_x11.c) with xMeM’s patches for [Android Hardware Buffers](https://github.com/leegao/mesa-wrapper-CI/blob/05bca95965ccebfeb2d34e415f5c5919e790a875/src/vulkan/wsi/wsi_common_android.c)) fails to initialize an externally sharable swapchain image, causing every device to fallback to the slower blit mode using a local primary and shared external secondary buffer.


* Certain Mali (and Adreno) devices only support a much older ion-heap UAPI, which causes the ion-heap import path to crash on these devices.



---

## Swapchain? Blit? ahardwarebuffers?



A swapchain is a (chain of) images that the game/application renders/scans-out to. Typically, your window system will keep a queue (chain) of these rendered frame images (e.g. to allow for better frame pacing, or to parallelize the display (presentation) of one frame from the rendering of the next).

In an ideal world, your game would have a direct handle of the image that will be presented on the next frame. Once it’s done drawing that frame, it just tells your compositor/display server, who also holds the same direct handle to that finished image, to display it to the user. Since the server also holds the same image (the physical data), it doesn’t need to perform a potentially expensive `CopyImage` operation every frame, hence saving time.

Abstractly, the handle of an image is often just a `VkImage` resource handle. Concretely, these handles are often backed by device (GPU accessible) memory. On Android, the concrete physical memory interface that backs externally sharable and device-visible memory are [Android Hardware Buffers](https://developer.android.com/ndk/reference/group/a-hardware-buffer) (or AHB).

```
==========================================================================================
DIRECT RENDERING PATH (Zero-Copy)
==========================================================================================

+-------------------+       BGRA8 View       +-----------------------+      AHB Import      +----------------+
|  Game / Renderer  | ---------------------> |   Shared RGBA8 AHB    | <------------------  | Display Server |
| (Virtual Format)  |                        |  (Physical Memory)    |   (Shared Handle)    |   (Winlator)   |
+-------------------+                        +-----------------------+                      +----------------+
                                                         ^
                                                         |
                                             Render Frame (No Copy Needed)
```

Earlier, I said that in an ideal world, your game (the renderer) and your display server (the compositor/presenter) share the same physical resource handle (externally shared memory). This is called direct rendering (DR). Unfortunately, due to certain external memory sharing constraints (e.g. on specific image formats not being sharable, or else on specific backing image types not being suitable for the swapchain image), this is not always possible.

When there are no suitable image format/memory pairs to support direct rendering, we have to fall back to an indirect rendering path known as **blitting**. Under this regime, the window system maintains a pair of images for every virtual swapchain image:

1. The **primary image**, a local (not shared) image that the game (renderer) directly renders to, and


2. The **secondary image**, an exported image that is guaranteed to be backed by externally shared memory, but may lack some features of the primary image required for the game to actually directly render to this image.



In particular, the *primary image* must support all of the properties/features required for the game engine to actually render to it (usually, this is a particular swapchain format such as `B8G8R8A8`, as well as a specific set of usage flags, such as the ability to be attached as a render color input attachment), which on some devices disqualifies it for external memory sharing. As a result, we create a secondary image that is solely responsible for external memory sharing, and we’ll synchronize the two by "**blitting**" (a fancy word for "copying") the data from the local primary image into the externally shared secondary image.


```
==========================================================================================
INDIRECT BLITTING PATH (One-Copy Fallback)
==========================================================================================

+-------------------+                        +-----------------------+
|  Game / Renderer  | ---------------------> |  Local Primary Image  |
|  (B8G8R8A8_UNORM) |      Renders To        |    (Local Memory)     |
+-------------------+                        +-----------------------+
                                                         |
                                                         | vkCmdCopyImage() [1-Copy Blit]
                                                         v
                                             +-----------------------+      AHB Import      +----------------+
                                             | Shared Secondary Image| <------------------  | Display Server |
                                             |  (R8G8B8A8 / AHB)     |   (Shared Handle)    |   (Winlator)   |
                                             +-----------------------+                      +----------------+
==========================================================================================
```

---

## Technical Requirements



Unfortunately, we don’t have unlimited "degrees of freedom" for the swapchain image formats that can be supported. As we will soon see, we can *only* advertise support for `BGRA8` swapchains, and this surprisingly leaves no options for direct rendering on certain (Mali) drivers.

### Winlator "X11" and B8G8R8A8



Prior to [the DisplayX renderer on Winlator Ludashi (see commit a1dfbde)](https://github.com/Pipetto-crypto/winlator/commit/a1dfbdec1f7cc7e09a1a1fae0bdbf245506fb23d#diff-e3a40c38780b91e3fcdb3a32afd9fbc7bf442f3a120aa2c5af76a62a0b92063bL480), the Winlator X11 emulated display server (as well as nearly every other Winlator-based emulators’ display servers) hardcode the expected swapchain format to `HAL_PIXEL_FORMAT_BGRA_8888`. This traces back to the original Winlator.

> **NOTE**: The upstream Winlator has actually [added support for R8B8G8A8 swapchain images (see commit 99e8e07)](https://github.com/brunodev85/winlator-app/commit/99e8e0709952a7326d73f9adf7bc617d461b6078) about a year ago, but most major forks of Winlator branched well before this change.

For our purpose (the M0 of this workaround), we’ll treat this as a static requirement that our final exported swapchain image to the Winlator display server MUST be an AHB in `B8G8R8A8` format.

Since the wrapper aims to work generally across all emulators, we’re also limited by the scope of what we can touch/fix. We can’t change the renderer in the emulator, only play along their rules.

This then constrains the (primary) swapchain image format supported by the wrapper [to just VK_FORMAT_B8G8R8A8](https://github.com/leegao/mesa-wrapper-CI/blob/wrapper-25/src/vulkan/wsi/wsi_common_x11.c#L446).

> **NOTE**: As a followup, a direct passthrough path for RGBA display servers for DisplayX (and other renderers that support RGBA swapchains) will be needed.

### Mali Drivers CANNOT export HAL_PIXEL_FORMAT_BGRA_8888



A second pin to constrain our problem is the fact that, unlike Adreno, Mali does not support importing `HAL_PIXEL_FORMAT_BGRA_8888` (which the winlator x11 display server expects) into Vulkan. We’ll just treat this as another hard requirement.

---

This already tells us why direct rendering fails on Mali devices. In particular, the wrapper (Mesa WSI + AHB backend) performs the [following probe](https://github.com/leegao/mesa-wrapper-CI/blob/wrapper-25/src/vulkan/wsi/wsi_common_android.c#L21,L39) to check for DR support:

```c
if (AHardwareBuffer_allocate(&(AHardwareBuffer_Desc){
        .width = 500,
        .height = 500,
        .layers = 1,
        .format = AHARDWAREBUFFER_FORMAT_B8G8R8A8_UNORM,
        .usage = AHARDWAREBUFFER_USAGE_GPU_FRAMEBUFFER |
                 AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE |
                 AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
                 AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN
    }, &ahardware_buffer) != 0) {
    WRAPPER_LOG(error, "Failed to allocate ahardware buffer, blitting");
    return WSI_SWAPCHAIN_IMAGE_BLIT;
}

VkAndroidHardwareBufferFormatPropertiesANDROID ahardware_buffer_format_props = {
    .sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID,
    .pNext = NULL,
};

VkAndroidHardwareBufferPropertiesANDROID ahardware_buffer_props = {
    .sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID,
    .pNext = &ahardware_buffer_format_props,
};

result = wsi->GetAndroidHardwareBufferPropertiesANDROID(
    device, ahardware_buffer, &ahardware_buffer_props);

```

While the `AHardwareBuffer_allocate` succeeds (ARM devices are still capable of *allocating* `BGRA_8888` AHBs), importing them into Vulkan fails (the `vkGetAndroidHardwareBufferPropertiesANDROID`).

Regardless, we would need to allocate the following swapchain image:

```c
VkExternalMemoryImageCreateInfo ext_mem_info = {
    .sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
    .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID,
};

VkImageCreateInfo no_blit_image_info = {
    .sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
    .pNext       = &ext_mem_info,
    .flags       = VK_IMAGE_CREATE_ALIAS_BIT,
    .format      = VK_FORMAT_B8G8R8A8_UNORM,
    .extent      = requested_extent,
    .tiling      = VK_IMAGE_TILING_OPTIMAL,
    .usage       = requested_usage,
    /* ... mipLevels, arrayLayers, etc. ... */
};

```

This would have failed anyways, as importing BGRA AHBs is disallowed on the Mali driver. Instead, we fall back to the blit path.

---

## The Blitting Path



To make blitting work for mobile drivers where certain image formats (like BGRA) cannot be exported, xMeM came up with an ingenious hack. Just lie to the driver and say our image handle used to pass data between the renderer (the game) and the display server (Winlator) is always in `R8B8G8A8` mode instead, even if physically, the data contained in these buffers are still in `BGRA_8888` order.

This is typically not fully safe in specific circumstances where the driver can be *too* smart for its own good. In particular, when calling `vkCmdBlitImage` from a `RGBA_8888` image to a `BGRA_8888` image, Vulkan will automatically swizzle the data order. Fortunately, the entire blit lives internally within the WSI engine, so we can simply choose to never use the unsafe `vkCmdBlitImage` (despite the name of this operation).

The full solution that the WSI wrapper performs is as follows:

* The application requests a `BGRA_8888` swapchain image.


* The WSI engine detects that `BGRA_8888` swapchain images are not importable from AHB, so it switches to blit mode.


* The WSI creates a **local** (not exportable) primary image in `BGRA_8888` mode:


```c
VkImageCreateInfo primary_image_info = {
    .sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
    .pNext       = NULL, // -> Local memory
    .flags       = VK_IMAGE_CREATE_ALIAS_BIT,
    .format      = VK_FORMAT_B8G8R8A8_UNORM,
    .extent      = requested_extent,
    .tiling      = VK_IMAGE_TILING_OPTIMAL,
    .usage       = requested_usage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, // needs to copy to a secondary buffer
    /* ... mipLevels, arrayLayers, etc. ... */
};

```


* The WSI returns this local primary image to the game as the render surface.


* The WSI then creates a second **external** (exportable via AHB) secondary image in `RGBA_8888` mode:


```c
VkExternalMemoryImageCreateInfo secondary_ext_mem_info = {
    .sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
    .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID,
};

VkImageCreateInfo secondary_image_info = {
    .sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
    .pNext       = &secondary_ext_mem_info, // -> External AHB
    .flags       = 0u,
    .format      = VK_FORMAT_R8G8B8A8_UNORM, // valid configuration for AHB
    .extent      = requested_extent,
    .tiling      = VK_IMAGE_TILING_LINEAR, // -> Linear for export
    .usage       = VK_IMAGE_USAGE_TRANSFER_DST_BIT, // needs to be copied to from primary buffer
    /* ... mipLevels, arrayLayers, etc. ... */
};

```


* Finally, when present is called, a [`vkCmdCopyImage`](https://github.com/leegao/mesa-wrapper-CI/blob/05bca95965ccebfeb2d34e415f5c5919e790a875/src/vulkan/wsi/wsi_common.c#L2091) is issued to perform a one-copy "blit" (as opposed to the zero-copy under DR mode) from the primary render buffer to the secondary export buffer that is exported to the display server.



In particular:

* When the game renders a frame to its swapchain (that is requested in `[B, G, R, A]` order), it will physically write `[B, G, R, A]` texel data into this virtual *primary* `VK_FORMAT_B8G8R8A8` `VkImage` backed by local memory (valid configuration, as this `VkImage` in BGRA order is not backed by an AHB, which would be an invalid configuration on Mali).


* Then, the WSI engine copies the `[B, G, R, A]` texel data from the primary `VK_FORMAT_B8G8R8A8` `VkImage`, **byte for byte**, onto the secondary `VK_FORMAT_R8G8B8A8` `VkImage`, which now contains physical texel data laid out in the (mismatching) `[B, G, R, A]` order.


* If we were to now do anything in Vulkan in our process with this `VkImage`, it would have a swapped Blue/Red channel. Fortunately, we don’t do anything with this `VkImage` beyond exporting its data across process boundaries.




* The secondary `VK_FORMAT_R8G8B8A8` `VkImage` is a valid configuration for AHB external memory, even if it physically contains `[B, G, R, A]` ordered data, which is an application-level contract. As a result, the Winlator display server process can also grab a direct handle to this same buffer memory (via AHB).


* Finally, the Winlator X11 display server picks up this memory handle containing physical `[B, G, R, A]` frame data, imports it correctly as a `AHARDWAREBUFFER_FORMAT_B8G8R8A8_UNORM` buffer, and proceeds to display this image correctly.



The main workaround here is that "IPC" between the renderer and the display server:

* The renderer creates a `RGBA_8888` (valid in Mali) external buffer memory that contains physical texel data in `[B, G, R, A]` format.


* The display server wraps the same buffer memory as a `BGRA_8888` (which is also in the correct color order) to present it.



---

## What Breaks on Mali?



Two major bugs occur in the Mali driver with this implementation today.

### Direct Rendering always fails



Because the probe tries to import an `AHARDWAREBUFFER_FORMAT_B8G8R8A8_UNORM` buffer into Vulkan (illegal in the Mali driver), it always fails, causing a fallback to the blit path.

This causes a slight performance slowdown since we need an extra copy per present.

### Blitting fails for 32-bit VK_EXT_map_memory_placed emulation



The primary image (the BGRA render target) is created like so in WSI:

```text
Calling wsi->CreateImage(chain->device, &info->create, &chain->alloc, &image->image)
  .flags: VkImageCreateFlags = 0x400
  .imageType: VkImageType = 0x1
  .format: VkFormat = 0x2c // VK_FORMAT_B8G8R8A8_UNORM
  .extent: VkExtent3D
    .width: uint32_t = 0x500
    .height: uint32_t = 0x2d0
    .depth: uint32_t = 0x1
  .mipLevels: uint32_t = 0x1
  .arrayLayers: uint32_t = 0x1
  .samples: VkSampleCountFlagBits = 0x1
  .tiling: VkImageTiling = 0x0 // TILING_OPTIMAL
  .usage: VkImageUsageFlags = 0x13
  .sharingMode: VkSharingMode = 0x0
  .queueFamilyIndexCount: uint32_t = 0x1
  .pQueueFamilyIndices[0]: uint32_t* = 0x0
  .initialLayout: VkImageLayout = 0x0
  .pNext(1000001002) // MESA specific extension

```

This is exactly what we expect: there’s no `VkExternalMemoryImageCreateInfo` `pNext` extension for this image, so it should be local-only.

Except … (duh, something is broken here).

#### Wine-aarch64, syswow64, and 32-Bit Vulkan support



Before we look into what happens on Mali, I’ll catch you up on how 32-bit PC emulation works.

Wine-aarch64 by default, as the name suggests, runs in 64-bit mode. To interface with x86 code (both x86-64 and 32-bit i86), Wine uses a pair of binary translator plugins:

1. **For x86-64**, it needs an `arm64ec` translation plugin, which is most commonly `libarm64ecfex.dll`. For x86-64 `.exe`s and `.dll`s, Wine will delegate to this plugin to translate the x86-64 ISA object code into native arm64 object code, specifically using the Windows standard `arm64ec` ABI instead of the typical `aarch64` ABI used by Linux/Android. I won’t dive deeper on this topic here, but it is an impressive feat of engineering that allows hybrid `arm64`-`aarch64` native `.dll`s to run with mixed-mode x86-64 + `arm64ec` `.dll`s in the same process via this ABI hack.


2. **For 32-bit i86**, it needs a `syswow64` translation plugin, which despite its name, is an i86-32bit to aarch64 binary translator. On Wine, this can be either `box32` or `fexcore`. Since the target ABI is aarch64, the plugin interface is much simpler, hence the availability of more plugins that conform to this interface and ABI.



Under `syswow64` (32-bit) emulation, the guest application/game expects all memory handles it receives to fit into a 32-bit handle. Where Vulkan is concerned, this means that when you call:

```c
    vkMapMemory(2KHR)(...)
```

The handle returned by your unified 64-bit Vulkan driver **MUST** also fit into a 32-bit handle. Except, Vulkan does not do this at all. By default, unless you specifically specify where to map your memory, Vulkan will almost certainly return a 64-bit handle, which would crash the guest 32-bit application/game.

Fortunately for us, Wine actually handles this case fairly gracefully on your behalf. *If* your Vulkan driver supports the `VK_EXT_map_memory_placed` extension, then the Wine-vulkan hooks will automatically translate all `vkMapMemory` API calls into equivalent `vkMapMemory2KHR` ones that explicitly ask Vulkan to map the memory at a low (<32-bit) virtual address, which Wine-vulkan will then return back to the guest 32-bit process as a valid 32-bit handle.

Notice the caveat - ***if*** your driver supports `VK_EXT_map_memory_placed`. That’s because most mobile drivers, including Mali drivers, [do not support this extension](https://vulkan.gpuinfo.org/displayextensiondetail.php?extension=VK_EXT_map_memory_placed&platform=android) with the exception of Mesa (Turnip) drivers and recent Qualcomm blob drivers (v842 onwards).

#### VK_EXT_map_memory_placed Emulation in the Wrapper



To solve this problem, the wrapper specifically adds an emulation path for `VK_EXT_map_memory_placed` on drivers that do not support it outright (basically every Mali driver on Android).

The idea is actually quite elegant and simple:

Treat every memory allocation (via a hook at [`wrapper_AllocateMemory`](https://github.com/xMeM/mesa/blob/e65c7eb6ee2f9903c3256f2677beb1d98464103f/src/vulkan/wrapper/wrapper_device_memory.c#L294)) for host-visible memory (memory that can be `vkMapMemory`-ed) as external memory.

Why external memory? Because the external memory interface in Vulkan allows these to be exported to raw file descriptors. These file descriptors can then be primitively `mmap`-ed generically, which allows us to map these shared memory blocks at a fixed address using the flag [`MAP_SHARED | MAP_FIXED`](https://github.com/xMeM/mesa/blob/e65c7eb6ee2f9903c3256f2677beb1d98464103f/src/vulkan/wrapper/wrapper_device_memory.c#L436), exactly the semantics of `VK_EXT_map_memory_placed`.

This design is implemented in [`wrapper_device_memory.c`](https://github.com/xMeM/mesa/blob/wrapper/src/vulkan/wrapper/wrapper_device_memory.c) and inherited by all major forks of the wrapper.

One specific implementation detail here is how [`wrapper_AllocateMemory`](https://github.com/xMeM/mesa/blob/e65c7eb6ee2f9903c3256f2677beb1d98464103f/src/vulkan/wrapper/wrapper_device_memory.c#L294) actually picks what type of external memory to use:

```c
result = wrapper_allocate_memory_dmabuf(device, pAllocateInfo,
                                        pAllocator, &mem->dispatch_handle, &mem->dmabuf_fd);
if (result != VK_SUCCESS) {
    wrapper_device_memory_reset(mem);
    result = wrapper_allocate_memory_dmaheap(device,
                                             pAllocateInfo, pAllocator, &mem->dispatch_handle, &mem->dmabuf_fd);
}
if (result != VK_SUCCESS) {
    wrapper_device_memory_reset(mem);
    result = wrapper_allocate_memory_ahardware_buffer(device,
                                                      pAllocateInfo, pAllocator, &mem->dispatch_handle, &mem->ahardware_buffer);
}

```

It first tries to use the generic [`VK_EXT_external_memory_dma_buf`](https://vulkan.gpuinfo.org/displayextensiondetail.php?extension=VK_EXT_external_memory_dma_buf) / [`VK_KHR_external_memory_fd`](https://vulkan.gpuinfo.org/displayextensiondetail.php?extension=VK_KHR_external_memory_fd) path, which is generically [supported on Mali since r32p1](https://github.com/leegao/compat_layer/blob/ab75fd18f6edc415fee0eb2751135a1317b38a4d/README.md#driver-3210-45-gpu-profiles-5-issues).

With one caveat (which I cannot find in the Mali erratas): if the `VkMemoryAllocateInfo` contains a `pNext` `VkMemoryDedicatedAllocateInfo` (`VK_KHR_dedicated_allocation`), then this path will fail unless the image/buffer was created with an associated external memory handle type that matches DMA buf / opaque-fds. This seems to have been fixed at *some* point between r32p1 and r54p1.

Next, it’ll try to use `dmabuf-heap` or `ion-heap` to directly allocate a system-memory shared–mem FD, and then import it directly. As before, the [`VK_KHR_external_memory_fd`](https://vulkan.gpuinfo.org/displayextensiondetail.php?extension=VK_KHR_external_memory_fd) extension is supported since r32p1 generically.

Another caveat: on GKI 4.X, `dmabuf-heap` is not available, so this mechanism will fallback to using `ion-heap`. Unfortunately, there are a few (not mutually compatible) UAPIs for `ion-heap`, and certain Mali devices will use a kernel module with a UAPI that’s incompatible with the xMeM implementation (using legacy UAPI), and this too needs to be fixed for these devices to function correctly.

Finally, it will try to use AHB, which is generally supported on Android since almost forever (predates Vulkan) and is the most robust/well supported shared external memory API.

> **Note on the caveat above**: If `vkAllocateMemory` is called with `VK_KHR_dedicated_allocation`, on certain versions of the Mali driver, it is mutually exclusive with external memory allocation (`VK_KHR_external_memory_fd`, `VK_EXT_external_memory_dma_buf`, and `VK_ANDROID_external_memory_android_hardware_buffer`). This is surprising, because it is (was) not spec-compliant.
>
>

#### Except …



Let’s come back to why blitting fails for some range of Mali devices when combined with `VK_EXT_map_memory_placed` emulation. There are actually two separate failure points that lead to issues on, specifically, certain pre-GKI-android12 (<5.10) and older drivers.

First, remember when we said that the primary image should be allocated on a device-local memory heap? This is because we don’t ever need to map this memory back onto the CPU (host) to read/write it.

Turns out, Android uses a unified memory address space (common for integrated GPU setups) where the GPU (device) and the CPU (host) memory spaces are the exact same memory space. As a result, it’s common for every memory heap type to report themselves as *both* `DEVICE_LOCAL` and `HOST_VISIBLE`.

Unfortunately, because we can’t tell ahead of time if a block of memory allocated on a `HOST_VISIBLE` memory heap can in fact be host-mapped, this triggers the [`VK_EXT_map_memory_placed` emulation](https://github.com/leegao/mesa-wrapper-CI/blob/05bca95965ccebfeb2d34e415f5c5919e790a875/src/vulkan/wrapper/wrapper_device_memory.c#L329) even though it’s never used:

```c
VKAPI_ATTR VkResult VKAPI_CALL
wrapper_AllocateMemory(VkDevice _device,
                       const VkMemoryAllocateInfo* pAllocateInfo,
                       const VkAllocationCallbacks* pAllocator,
                       VkDeviceMemory* pMemory) {
   VK_FROM_HANDLE(wrapper_device, device, _device);
   struct wrapper_device_memory *mem;
   VkResult result;
   VkMemoryPropertyFlags property_flags =
      device->physical->memory_properties.memoryTypes[
         pAllocateInfo->memoryTypeIndex].propertyFlags;

   if (!(property_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
      goto fallback;

   if (!device->vk.enabled_features.memoryMapPlaced ||
       !device->vk.enabled_extensions.EXT_map_memory_placed)
      goto fallback;

   if (vk_find_struct_const(pAllocateInfo, IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID))
      goto fallback;
   if (vk_find_struct_const(pAllocateInfo, IMPORT_MEMORY_FD_INFO_KHR))
      goto fallback;
   if (vk_find_struct_const(pAllocateInfo, EXPORT_MEMORY_ALLOCATE_INFO))
      goto fallback;

   WRAPPER_LOG(info, "Emulating vkAllocateMemory");
   // ...
}

```

This then triggers a crash depending on the driver / GKI version:

##### 1. Incompatibility with VK_KHR_dedicated_allocation



One additional thing that our WSI does for blitting is that it’ll still create the primary image with [dedicated memory](https://github.com/leegao/mesa-wrapper-CI/blob/05bca95965ccebfeb2d34e415f5c5919e790a875/src/vulkan/wsi/wsi_common_android.c#L277) despite treating this as device-local memory. Whether this is intentional is unclear, but, as the primary image is not created with external memory, this extra `VkMemoryDedicatedAllocateInfo->image` handle is created with a device local external memory handle type (0).

Typically, even if AHB doesn’t support the underlying image format type (`BGRA_8888`), `dmabuf` *shouldn’t* have the same issue, as it’s just raw system memory without a specific layout. As a result, the call to [`wrapper_allocate_memory_dmabuf`](https://github.com/xMeM/mesa/blob/e65c7eb6ee2f9903c3256f2677beb1d98464103f/src/vulkan/wrapper/wrapper_device_memory.c#L139) should succeed. Unfortunately, due to the driver bug mentioned above, the presence of the `VK_KHR_dedicated_allocation` (`VkMemoryDedicatedAllocateInfo`) extension in `vkAllocateImage` actually causes the driver to fail/crash the allocation.

As a result, the memory allocation for the device-local primary swapchain image fails under 32-bit games due to the buggy interaction with the `VK_EXT_map_memory_placed` emulation in the wrapper.

##### 2. AHB getNativeHandle can return multiple (non-data) handles



In the actual [MemoryMap emulation](https://github.com/leegao/mesa-wrapper-CI/blob/wrapper-25/src/vulkan/wrapper/wrapper_device_memory.c#L464), when an AHB is attached to the memory, the call to `AHardwareBuffer_getNativeHandle` can return multiple file descriptors (some of which are not data handles). On certain Mali drivers, this does actually happen, and `handle->data[0]` is not always the data FD that can be `mmap`-ed.

##### 3. [Minor Bug] Ion-heap UAPI mismatch



On pre GKI-android12-5.10 devices, `dmabuf-heap` is absent. Instead, system memory allocations must be allocated on `/dev/ion` instead.

Under this regime, [`wrapper_allocate_memory_dmabuf`](https://github.com/xMeM/mesa/blob/e65c7eb6ee2f9903c3256f2677beb1d98464103f/src/vulkan/wrapper/wrapper_device_memory.c#L139) immediately fails (because of a lack of `dmabuf`). Next, [`wrapper_allocate_memory_dmaheap`](https://github.com/xMeM/mesa/blob/e65c7eb6ee2f9903c3256f2677beb1d98464103f/src/vulkan/wrapper/wrapper_device_memory.c#L89) attempts to allocate ion-heap memory:

```c
static int
ion_heap_alloc(int heap_fd, size_t size) {
   struct ion_allocation_data {
      __u64 len;
      __u32 heap_id_mask;
      __u32 flags;
      __u32 fd;
      __u32 unused;
   } alloc_data = {
      .len = size,
      /* ION_HEAP_SYSTEM | ION_SYSTEM_HEAP_ID */
      .heap_id_mask = (1U << 0) | (1U << 25),
      .flags = 0, /* uncached */
   };
   if (safe_ioctl(heap_fd, _IOWR('I', 0, struct ion_allocation_data),
                  &alloc_data) < 0)
      return -1;
   return alloc_data.fd;
}

```

However, there are actually 2 major, mutually incompatible UAPIs for ion-heap.

Prior to Linux 4.12, the ionheap kernel driver uses the [legacy UAPI](https://github.com/GameNative/mesa/pull/5/changes#diff-660f29f9ee3dd97cd63da99119f2ad822c6fde63307f24bd86c996dabbb26552R40), while the "modern" UAPI (used here in the wrapper) is available on GKI 4.14 onwards. However, Mali’s GKI seems to have decided against the ionheap modernization update in their GKI 4.12, 4.14, and 4.19, and instead only supports legacy UAPI.

As a result, non-AHB external memory (ionheap) immediately fails on devices without the GKI-android12-5.10 (mandatory `dmabuf` migration) kernel.

---

## Solution Sketches



### [Blitting] Skip VK_EXT_map_memory_placed emulation for swapchain images



Since the swapchain image is almost never actually host-mapped, a very simple and straightforward/safe fix is to just track if an image is the primary swapchain image created by our WSI engine. If so, just skip the map memory placed emulation path to avoid all of these problems. This is a very simple fix to implement.

* See commit [ffe2127](https://github.com/GameNative/mesa/pull/6/commits/ffe2127566a57887e22619cb1e94fa361e22a7d9) and commit [3653382](https://github.com/GameNative/mesa/pull/6/commits/3653382907a4e5f4e580ce4a9d1eaad57016ca59)


### [Direct Rendering] Emulate bgra swapchain image using a physical rgba AHB



To enable direct rendering support on Mali (and also indirectly bypass the blitting bugs as well), we can also try to emulate a virtual bgra-ordered swapchain image buffer using the same trick that our blitting path uses - by creating a `RGBA_8888` AHB buffer, importing it into Vulkan (this is legal with Mali drivers), and then "lying" to the application that our swapchain format is *really* a `VK_FORMAT_B8G8R8A8_UNORM` `VkImage(View)`.

This is actually very simple to do, with some caveats that rarely apply to swapchain images. Vulkan abstracts *most* image-related operations through `VkImageView`s. One of the cool things you can do with a `VkImageView` is to create one with a BGRA8 format (for the image *view*) on top of a physical RGBA8 `VkImage`. When the application performs image-view operations (using the BGRA8 view), the driver ensures that the read/write order (relative to the physical memory) conforms to our expected `[B, G, R, A]` order.

The caveats are the exceptions where ImageViews are not used, and are [documented here](https://github.com/GameNative/mesa/pull/6) (in addition to the very tricky case where a compute shader does direct memory modifications/accesses from the frame buffer without using `texel_fetch` primitives, which is exceedingly rare).

* See commit [2af0e2e](https://github.com/GameNative/mesa/pull/6/commits/2af0e2e588792a40fd092f73f64da0ff1b9fcb35)


### [Bug] Unlink VK_KHR_dedicated_allocation structs



`VK_KHR_dedicated_allocation` is a non-critical hint to the driver, so to avoid driver crashes, we can simply unlink this hint.

* See [PR #8 Changes](https://github.com/GameNative/mesa/pull/8/changes/0bd55cbed77ddade643b31646dad69db8f04f08f)


### [Bug] lseek every AHB handle fd to determine the right one to mmap



To avoid the `VK_EXT_map_memory_placed` emulation failing on devices that return multiple AHB handles, just perform a basic `lseek` operation (to determine validity and data size) and skip ones that result in `ESPIPE`.

* See [PR #7 Changes](https://github.com/GameNative/mesa/pull/7/changes)


### [Bug] Add a version check to toggle between the correct ionheap ioctl uapi



Turnip actually has an implementation of this, so we just need to do the same.

* See [PR #5 Changes](https://github.com/GameNative/mesa/pull/5/changes)
