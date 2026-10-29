"""Capture physical host information; do not assume homogeneous cores."""
from datetime import datetime, timezone
import os
import platform
import subprocess
import sys
from common import STUDENT_ID, save_json


def sysctl(key):
    result = subprocess.run(['sysctl', '-n', key], text=True, capture_output=True)
    value = result.stdout.strip() if result.returncode == 0 else ''
    return int(value) if value.isdecimal() else value or None


def main():
    info = {'student_id': STUDENT_ID, 'captured_at_utc': datetime.now(timezone.utc).isoformat(),
            'platform': platform.platform(), 'os': platform.system(), 'kernel': platform.release(),
            'machine': platform.machine(), 'logical_cores': os.cpu_count(),
            'cpu_model': platform.processor(), 'physical_cores': None,
            'python': sys.version, 'python_executable': sys.executable,
            'multiprocessing_start_method': 'spawn', 'cpu_affinity': 'OS scheduled, no pinning',
            'note': 'Real local host. Other OS activity and power/thermal state are uncontrolled.'}
    if platform.system() == 'Darwin':
        info.update(cpu_model=sysctl('machdep.cpu.brand_string'), physical_cores=sysctl('hw.physicalcpu'),
                    logical_cores=sysctl('hw.logicalcpu'), memory_bytes=sysctl('hw.memsize'),
                    cache_line_bytes=sysctl('hw.cachelinesize'))
        info['core_groups'] = [{key: sysctl(f'hw.perflevel{i}.{key}')
                               for key in ('name', 'physicalcpu', 'logicalcpu', 'l1dcachesize', 'l2cachesize')}
                              for i in range(sysctl('hw.nperflevels') or 0)]
    save_json('host_profile.json', info)
    print(info)


if __name__ == '__main__':
    main()
