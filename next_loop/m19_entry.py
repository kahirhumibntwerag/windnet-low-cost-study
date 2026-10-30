"""Execute M19 on a remote worker with the frozen M18 evidence volume."""
import subprocess
import sys


def main():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet',
                           'numpy==2.1.2', 'scipy==1.16.1', 'scikit-learn==1.7.2',
                           'requests==2.32.3', 'joblib==1.5.2'])
    subprocess.check_call([sys.executable, '-m', 'unittest', 'next_loop.test_m19_kernel',
                           'next_loop.test_regions', 'next_loop.test_topology',
                           'next_loop.test_target_weights'])
    from next_loop.m19 import run
    run()


if __name__ == '__main__': main()
