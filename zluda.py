import torch
import triton
import triton.language as tl

zluda_device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "[ZLUDA Device Not Found - Assuming ZLUDA]"

MEM_BUS_WIDTH = {
    "AMD Radeon RX 9070 XT": 256,
    "AMD Radeon RX 9070": 256,
    "AMD Radeon RX 9060 XT": 192,
    "AMD Radeon RX 7900 XTX": 384,
    "AMD Radeon RX 7900 XT": 320,
    "AMD Radeon RX 7900 GRE": 256,
    "AMD Radeon RX 7800 XT": 256,
    "AMD Radeon RX 7700 XT": 192,
    "AMD Radeon RX 7700": 192,
    "AMD Radeon RX 7650 GRE": 128,
    "AMD Radeon RX 7600 XT": 128,
    "AMD Radeon RX 7600": 128,
    "AMD Radeon RX 7500 XT": 96,
    "AMD Radeon RX 6950 XT": 256,
    "AMD Radeon RX 6900 XT": 256,
    "AMD Radeon RX 6800 XT": 256,
    "AMD Radeon RX 6800": 256,
    "AMD Radeon RX 6750 XT": 192,
    "AMD Radeon RX 6700 XT": 192,
    "AMD Radeon RX 6700": 160,
    "AMD Radeon RX 6650 XT": 128,
    "AMD Radeon RX 6600 XT": 128,
    "AMD Radeon RX 6600": 128,
    "AMD Radeon RX 6500 XT": 64,
    "AMD Radeon RX 6400": 64,
}

do_nothing = lambda _: None
def do_hijack():
    _get_props = triton.runtime.driver.active.utils.get_device_properties
    def patched_props(device):
        props = _get_props(device)
        name = torch.cuda.get_device_name()[:-8]  # Remove [ZLUDA]
        props["mem_bus_width"] = MEM_BUS_WIDTH.get(name, 128)
        if name not in MEM_BUS_WIDTH:
            print(f'  ::  Using default mem_bus_width=128 for {name}')
        return props
    triton.runtime.driver.active.utils.get_device_properties = patched_props
    from flash_attn.flash_attn_triton_amd import interface_fa
    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    def amd_flash_wrapper(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        try:
            if (query.shape[-1] <= 128 and
                attn_mask is None and
                query.dtype != torch.float32):
                if scale is None:
                    scale = query.shape[-1] ** -0.5
                return interface_fa.fwd(
                    query.transpose(1, 2),
                    key.transpose(1, 2),
                    value.transpose(1, 2),
                    None, None, dropout_p, scale,
                    is_causal, -1, -1, 0.0, False, None
                )[0].transpose(1, 2)
        except Exception as e:
            print(f'  ::  Flash attention error during execution: {str(e)}')
        return original_sdpa(query=query, key=key, value=value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)

    torch.nn.functional.scaled_dot_product_attention = amd_flash_wrapper

    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp = do_nothing
    torch.backends.cudnn.enabled = True
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)

do_hijack()
