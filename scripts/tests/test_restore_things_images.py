import hashlib

import restore_things_images as rti


def make_manifest(files):
    return [{'file_name': name, 'things_path': f'images/x/{name}',
             'sha256': hashlib.sha256(content).hexdigest()} for name, content in files.items()]


def test_manifest_lists_29_unique_jpgs_with_sha256():
    manifest = rti.read_manifest()
    names = [row['file_name'] for row in manifest]
    assert len(names) == 29 == len(set(names))
    assert all(name.endswith('.jpg') for name in names)
    assert all(row['things_path'].startswith('images/') and row['things_path'].endswith('/' + row['file_name'])
               for row in manifest)
    # a SHA-256 digest is 64 hexadecimal characters
    assert all(len(row['sha256']) == 64 and int(row['sha256'], 16) >= 0 for row in manifest)


def test_manifest_images_are_not_tracked_but_ignored():
    gitignore = (rti.REPO_ROOT / '.gitignore').read_text()
    for row in rti.read_manifest():
        assert row['file_name'] in gitignore


def test_sha256_of_known_content(tmp_path):
    path = tmp_path / 'a.jpg'
    path.write_bytes(b'abc')
    # SHA-256 test vector of 'abc' (FIPS 180-2)
    assert rti.sha256(path) == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'


def test_restore_copies_matching_and_reports_the_rest(tmp_path):
    source = tmp_path / 'things' / 'images'
    (source / 'bee').mkdir(parents=True)
    (source / 'bee' / 'bee_01b.jpg').write_bytes(b'bee')
    (source / 'bee' / 'bee_02s.jpg').write_bytes(b'changed')
    manifest = make_manifest({'bee_01b.jpg': b'bee', 'bee_02s.jpg': b'original', 'key_03s.jpg': b'key'})
    targets = (tmp_path / 'experiment', tmp_path / 'experiment_mid_lab')

    missing, mismatched = rti.restore(tmp_path / 'things', manifest, targets)

    assert missing == ['key_03s.jpg']
    assert mismatched == ['bee_02s.jpg']
    for target in targets:
        assert (target / 'bee_01b.jpg').read_bytes() == b'bee'
        assert not (target / 'bee_02s.jpg').exists()
