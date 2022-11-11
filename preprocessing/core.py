from pathlib import Path
import gzip
import json

from mypacket import MyPacket
from utils import read_pcap
from ENV import *

_NJOB = 2

class TransformPCAP:


    def __init__(self, path, path_output, batch_size=10000):

        self.path = path
        self.path_output = path_output
        self.str_path_output = str(path_output.absolute())
        self.batch_size = 10000
        self.rows = []

    def preocess(self):
        if Path(self.str_path_output + '_SUCCESS').exists():
            print(self.path_output, 'Done')
            return

        idx = 0
        for i, packet in enumerate(read_pcap(self.path)):
            mypacket = MyPacket(packet)
            arr = mypacket.transform()

            self._check_none(arr)

            if self.rows and (i > 0 and i%self.batch_size==0):
                
                self._write(idx)
                self.rows.clear()
                idx += 1

        if self.rows:
            self._writeidx

        with Path(self.str_path_output + '_SUCCESS').open('w') as f:
            f.write('') 

    def _check_none(self, arr):
        if arr is not None:
            # get labels for app identification
            prefix = self.path.name.split('.')[0].lower()
 
            row = {
                'app': PREFIX_TO_APP_ID.get(prefix),
                'traffic': PREFIX_TO_TRAFFIC_ID.get(prefix),
                'feature': arr.todense().tolist()[0]
            }
            self.rows.append(row)

    def _write(self, idx):
        p_output_path = Path(self.str_path_output + f'_part_{idx:04d}.json.gz')
    
        with p_output_path.open('wb') as f, gzip.open(f, 'wt') as f_out:
            for row in self.rows:
                f_out.write(f'{json.dumps(row)}\n')

    def _write_success(self):
         with Path(self.str_path_output + '_SUCCESS').open('w') as f:
            f.write('')


def main(data_dir, target_dir):
    data_dir = Path(data_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    for pcap_path in sorted(data_dir.iterdir()):
            transformer = TransformPCAP(pcap_path, target_dir / (pcap_path.name + '.transformed'))
            transformer.process()