from pathlib import Path

from mypacket import MyPacket
from utils import read_pcap


def preocess_pcap_file(path, path_output: Path = None, batch_size: int = 10000):
    if Path(str(path_output.absolute()) + '_SUCCESS').exists():
        print(path_output, 'Done')
        return

    