---
title: "Mali Adventures: Part 1 - DXVK 1.7.3 Bug and Op(Spec)CompositeConstant"
date: 2025-08-10
---

Disclaimer: I do not own a Mali device, nor have I reverse engineered the ARM blob drivers. These are just my hypotheses for the bug based on user bug reports.

## A Wild Mali Bug Appears in DXVK 1.7.3

See [https://github.com/leegao/bionic-vulkan-wrapper/issues/93](https://github.com/leegao/bionic-vulkan-wrapper/issues/93)

[@Artewar67](https://github.com/Artewar67) discovered something very interesting on their Mali-G76 device. They found that several games run fine on dxvk 1.7.2, but not on 1.7.3

So, after analyzing the differences between [v1.7.2...v1.7.3](https://github.com/doitsujin/dxvk/compare/v1.7.2...v1.7.3), we quickly narrowed down the potential issue down to:

1. [doitsujin/dxvk@45461ee](https://github.com/doitsujin/dxvk/commit/45461ee54e691739abfae2edc61771b359302119)
2. [doitsujin/dxvk@538b559](https://github.com/doitsujin/dxvk/commit/538b55921ecea72ef861d2b5da72ac20b109d2be)

which introduces this seemingly innocuous pattern into shaders:

```
%texture_0_bound = OpSpecConstantTrue %bool
...
    %float_0 = OpConstant %float 0
       %v4_0 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
     %v4bool = OpTypeVector %bool 4
%v4_t0_bound = OpConstantComposite %v4bool %texture_0_bound %texture_0_bound %texture_0_bound %texture_0_bound
...
         %41 = OpSampledImage %38 %40 %39
         %42 = OpImageSampleImplicitLod %v4float %41 %37
         %47 = OpSelect %v4float %v4_t0_bound %42 %v4_0
```
See [https://www.diffchecker.com/Vx945RKM/](https://www.diffchecker.com/Vx945RKM/) for a full example.

This injects an `is_texture/texel_bound` check in the form of an op-select of whether or not a `vec4(texture_0_bound)` is true, and if not, to default to a transparent color (`vec4(0)`) instead. This is effectively the following glsl block

```glsl
vec4 texel = texture_0_bound ? image_sample(...) : vec4(0);
```

This seems all safe and good, but as we saw earlier, this fails spectacularly on Mali for some reason. Let's dig into why.

### The Issue

(I believe that) the problem is that `%texture_0_bound` is a `OpSpecConstant`, which can be specialized (e.g. changed) by the pipeline creation itself. What I believe is happening on Mali is that the driver (for some range of versions) does not properly track the fact that these are not compile-time constants and can be re-specialized by the pipeline itself. Instead, it seems to be defaulting to False (even if it's a OpSpecConstantTrue), and then the OpSelect gets aggresively optimized away into the `vec4(0)` black color instead.

### The Fix

There seems to be two camps on how to properly workaround the driver shader compiler bug here.

#### Winlator

Bruno was kind enough to describe his algorithm for this problem as part of [brunodev85/winlator#1096 (comment)](https://github.com/brunodev85/winlator/issues/1096#issuecomment-3065756958)

```c
uint32_t instLength;
SpvOp opcode = SpvOpNop;
bool inspectOpSelect = false;

for (int i = CODE_START, size = codeSize / sizeof(uint32_t); i < size;) {
    opcode = SPV_INST_OPCODE(code[i]);
    instLength = SPV_INST_LENGTH(code[i]);

    if (opcode == SpvOpSampledImage || opcode == SpvOpImageFetch) {
        inspectOpSelect = true;
    }
    else if (opcode == SpvOpSelect && inspectOpSelect) {
        uint32_t* object1Id = &code[i+4];
        uint32_t* object2Id = &code[i+5];
        int index = fetchSpvInstIndex(code, codeSize, CODE_START, SpvOpConstantComposite, 2, *object2Id, true);

        uint32_t scalarId = index != -1 ? code[index+3] : *object2Id;
        index = fetchSpvInstIndex(code, codeSize, CODE_START, SpvOpConstant, 2, scalarId, true);
        if (index != -1) {
            uint32_t constValue = code[index+3];
            if (constValue == 0) *object2Id = *object1Id;
        }

        inspectOpSelect = false;
    }
    else if (opcode == SpvOpStore) {
        inspectOpSelect = false;
    }

    i += instLength;
}
```

the idea is essentially to ensure that the false branch of that OpSelect will also return the correct color (with the assumption that rarely will the texture be legitimately unbound, and potentially will have other safeguards to prevent an out-of-bounds there from causing a GPU fault). This will ensure that even the buggy shader compiler will still correctly (wrongly) optimize this shader to the correct final form that uses the real color instead of the `vec4(0)`.

#### GameFusion

If you look at the spirv dump of the finaly transformed shader within GameFusion, you'll notice that they too have a `spirv::rewriter::MaliCompositeConstantFixPass` class. However, the transformation it performs seems to be very different from Winlator:

```
%texture_0_bound = OpSpecConstantTrue %bool
...
    %float_0 = OpConstant %float 0
       %v4_0 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
     %v4bool = OpTypeVector %bool 4
%v4_t0_bound = OpSpecConstantComposite %v4bool %texture_0_bound %texture_0_bound %texture_0_bound %texture_0_bound ; <<<- changed from OpConstantComposite to OpSpecConstantComposite
...
         %41 = OpSampledImage %38 %40 %39
         %42 = OpImageSampleImplicitLod %v4float %41 %37
         %47 = OpSelect %v4float %v4_t0_bound %42 %v4_0
```

this single line change of the conditional variable from an OpConstantComposite to an OpSpecConstantComposite seems to be sufficient in avoiding the buggy shader compilation optimizations.

---

As far as I can tell, [ARM Mali Errata](https://documentation-service.arm.com/static/67ca1a5ece2747241fced502) does not seem to list this specific issue
