"""Put the 29 stimulus images taken from the THINGS database back into the experiment folders.

The THINGS licence does not allow redistributing its images, so they are not
part of this repository. Download the THINGS images yourself from
https://osf.io/jum2f/ (images_THINGS.zip, after agreeing to its licence),
unzip them, and run from `scripts/`:

    python restore_things_images.py <folder with the unzipped THINGS images>

The folder can be any folder that contains the image files, for example the
`images` folder of the archive (which holds one subfolder per concept). Every
image is checked against the SHA-256 checksum, listed in
`resources/things_images.csv`, of the file shown to the observers, and copied
into `experiment/images` and `experiment_mid_lab/images`.
"""
import csv
import hashlib
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / 'resources/things_images.csv'
TARGET_FOLDERS = (REPO_ROOT / 'experiment/images', REPO_ROOT / 'experiment_mid_lab/images')


def read_manifest(path=MANIFEST):
    """Rows of the manifest as dicts with keys file_name, things_path and sha256."""
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def find_image(source, file_name):
    """Path of `file_name` anywhere below `source`, or None."""
    matches = sorted(Path(source).rglob(file_name))
    return matches[0] if matches else None


def restore(source, manifest=None, targets=TARGET_FOLDERS):
    """Copy every manifest image found below `source` into the target folders.

    Returns the lists (missing, mismatched) of file names that could not be
    restored; nothing is copied for those.
    """
    manifest = read_manifest() if manifest is None else manifest
    missing, mismatched = [], []
    for row in manifest:
        found = find_image(source, row['file_name'])
        if found is None:
            missing.append(row['file_name'])
            continue
        if sha256(found) != row['sha256']:
            mismatched.append(row['file_name'])
            continue
        for target in targets:
            Path(target).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(found, Path(target) / row['file_name'])
    return missing, mismatched


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    missing, mismatched = restore(argv[1])
    for name in missing:
        print(f'not found: {name}')
    for name in mismatched:
        print(f'checksum differs from the file used in the experiment: {name}')
    n_restored = len(read_manifest()) - len(missing) - len(mismatched)
    print(f'restored {n_restored} images')
    return 0 if not (missing or mismatched) else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
