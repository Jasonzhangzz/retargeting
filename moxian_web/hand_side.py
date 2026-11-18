"""Validated hand selection; USB product IDs are only a discovery hint."""
CONFIRMED_DEVICE_SIDES={'2DF5F3B00DA5':'left'}

def check_side(side):
    if side not in ('left','right'):raise ValueError('手侧必须为 left 或 right')
    return side

def device_side(device):
    return check_side(CONFIRMED_DEVICE_SIDES.get(device['serial'],device['usb_side']))
