# Longshot's temporal upscalers

From Longshot (https://git.magicbean.dev/MagicBeanDev/Longshot), commit 544229a260c5f5008f241b9cf48ae6b01c1602b9,
`gpu/vk/`. GPL-3.0-or-later (LICENSE here is Longshot's). The chain is hckrman101's NVIDIA DLSS work (#841) as Longshot
generalised it; the FSR 3.1 passes are AMD's FidelityFX shaders (MIT) as Longshot built them.

Unchanged files: vk_temporal.{cpp,h}, vk_dlss.{cpp,h}, vk_fsr3.{cpp,h}, vk_fsr3_inputs.{cpp,h}, vk_fsr3fi.{cpp,h},
vk_fsr4.h, vk_xess.h, dlss_actor.h, dlss_checks.h, dlss_math.h, fsr3_math.h, temporal_jitter.h, upscaler_rows.h,
xess_rules.h, shaders/*.

`compat/` is this project's: headers standing in for the ones those files include from the rest of Longshot's renderer
(vk_internal.h and the rest). What they declare is defined in runtime/native/nvn/longshot_upscale.cpp on this renderer.

NVIDIA's DLSS SDK is not here and must not be added (NVIDIA's licence): see the CMake option OOT_NGX_SDK.
