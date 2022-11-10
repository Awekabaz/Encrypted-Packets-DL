from pathlib import Path

from scapy.layers.dns import DNS
from scapy.layers.inet import TCP
from scapy.packet import Padding
from scapy.utils import PcapReader

def read_pcap_file(path: Path) -> PcapReader:

    return PcapReader(str(path))


def should_omit_packet(packet)-> bool:
    
    if (packet.flags & 0x13) and (TCP in packet):
        # not payload OR there is only padding
        layers = packet[TCP].payload.layers()

        if not layers or (len(layers) == 1 and Padding in layers):
            return True

    # DNS segment
    if DNS in packet: return True

    return False