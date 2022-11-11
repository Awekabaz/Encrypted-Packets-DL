from pathlib import Path

from transform import MyPacket
from utils import read_pcap


def preocess_pcap_file(path, path_output: Path = None, batch_size: int = 10000):
