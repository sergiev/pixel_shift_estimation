import imreg_dft as ird

import struct
import numpy as np


MAV_OPTICAL_FLOW_message_id = 100
MAV_OPTICAL_FLOW_id = 0  # unused
MAV_OPTICAL_FLOW_extra_crc = 175
MAV_system_id = 1
MAV_component_id = 0x54
PACKET_ID = 0

# достаточно для корректной работы на 330м при <6000х4000 <10m/s 25+fps
WINDOW_SIZE = 30


# TODO получить от К
def is_flowhold() -> bool:
    """True if we shouldn't move, False otherwise"""
    pass


# TODO получить от К
def get_current_img() -> np.ndarray:
    """Get latest available image from sensor as numpy ndarray
    Should be grayscale!"""
    pass


def prepare_img(img: np.ndarray) -> np.ndarray:
    """Current image preprocessing"""
    return img[:WINDOW_SIZE, :WINDOW_SIZE]


def get_prepared_img() -> np.ndarray:
    return prepare_img(get_current_img())


# TODO узнать у К замену в новой архитектуре
def clock_tick() -> None:
    """OpenMV's clock.tick() analogue"""
    pass


# copied from openMV
# https://github.com/mavlink/c_library_v1/blob/master/checksum.h
def checksum(data, extra):
    output = 0xFFFF
    for i in range(len(data)):
        tmp = data[i] ^ (output & 0xFF)
        tmp = (tmp ^ (tmp << 4)) & 0xFF
        output = ((output >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xFFFF
    tmp = extra ^ (output & 0xFF)
    tmp = (tmp ^ (tmp << 4)) & 0xFF
    output = ((output >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xFFFF
    return output


def send_optical_flow_packet(x: int, y: int, quality: float) -> None:
    """Wrap and (TODO)send pixel shift estimation as OpenMV's optical flow packet"""
    global PACKET_ID
    temp = struct.pack(
        "<qfffhhbb", 0, 0, 0, 0, int(x), int(y), MAV_OPTICAL_FLOW_id, int(quality * 255)
    )
    temp = struct.pack(
        "<bbbbb26s",
        26,
        PACKET_ID & 0xFF,
        MAV_system_id,
        MAV_component_id,
        MAV_OPTICAL_FLOW_message_id,
        temp,
    )
    temp = struct.pack("<b31sh", 0xFE, temp, checksum(temp, MAV_OPTICAL_FLOW_extra_crc))
    PACKET_ID += 1
    # TODO uart.write(packet) or something equal


first = get_prepared_img()
while True:
    clock_tick()
    current = get_prepared_img()
    x, y, quality = 0, 0, 0
    if not is_flowhold():
        first = current
    else:
        shift = ird.translation(im0=first, im1=current)
        x, y = shift["tvec"]
        quality = ird["success"]
    send_optical_flow_packet(x, y, quality)
    print("{0:+f}x {1:+f}y quality: {2}".format(x, y, quality))
