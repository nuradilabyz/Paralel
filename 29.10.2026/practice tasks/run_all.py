"""Run the requested experiments sequentially and rebuild the answer report."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    (ROOT/'results').mkdir(exist_ok=True)
    scripts = [('profile_env.py', 'profile_console.txt'),
               ('exp_a_serial.py', 'experiment_a_console.txt'),
               ('exp_b_contention.py', 'experiment_b_console.txt'),
               ('exp_c_fixed.py', 'experiment_c_console.txt'),
               ('verify_results.py', 'verification_console.txt'),
               ('build_report.py', 'report_console.txt')]
    for script, log in scripts:
        print(f'Running {script}', flush=True)
        process = subprocess.Popen([sys.executable, '-u', str(ROOT/script)], cwd=ROOT,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        with (ROOT/'results'/log).open('w') as stream:
            for line in process.stdout:
                print(line, end='', flush=True)
                stream.write(line)
        if process.wait() != 0:
            raise SystemExit(f'{script} failed; inspect results/{log}')


if __name__ == '__main__':
    main()
