from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, UDP
from scapy.packet import Padding
from scapy.compat import raw

import numpy as np
from scipy import sparse

class MyPacket:

    def __init__(self, packet):

        self.packet = packet
        self.array = None
        self.metadata = None

    def transform(self):

        self._ether_header()
        self._pad_udp()
        self._ip()
        self._packet_to_arr()

        return self.array

    def _ether_header(self):
        
        if Ether in self.packet:
            self.packet =  self.packet[Ether].payload

    def _pad_upd(self):
        if UDP in self.packet:
            # get layers after and before udp
            layer_after =self.packet[UDP].payload.copy()
            layer_before = self.packet.copy()
            layer_before[UDP].remove_payload()

            # padding layer
            pad_layer = Padding()
            pad_layer.load = '\x00' * 12

            self.packet = layer_before / pad_layer / layer_after

    def _ip(self):
        if IP in self.packet:
            self.packet[IP].src = '0.0.0.0'
            self.packet[IP].dst = '0.0.0.0'


    def _packet_to_arr(self, max_length: int = 1500):
        array = np.frombuffer(raw(self.packet), dtype=np.uint8)[0: max_length]
        array /= 255

        if len(array) < max_length:
            adj_width = max_length - len(array)
            array = np.pad(array, pad_width=(0, adj_width), constant_values=0)

        self.array = sparse.csr_matrix(array)
        