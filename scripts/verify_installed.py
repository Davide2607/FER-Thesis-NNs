packages = [
    'tensorflow',
    'tensorflow_io',
    'h5py',
    'numpy',
    'PIL',
    'sklearn',
    'ultralytics',
    'gdown',
    'mediapipe',
    'torch'
]

def check(pkg):
    try:
        m = __import__(pkg)
        v = getattr(m, '__version__', None)
        # some packages expose version differently
        if v is None:
            v = getattr(m, 'version', None)
        print(f"{pkg}: OK, version={v}")
    except Exception as e:
        print(f"{pkg}: FAILED -> {e}")

if __name__ == '__main__':
    for p in packages:
        check(p)
