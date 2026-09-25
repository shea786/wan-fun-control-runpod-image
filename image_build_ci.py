"""Build a pinned Wan OCI image on GitHub's hosted runner, never the controller.

Downloads one 2 GB range at a time, appends it as a separate GHCR layer,
then discards local bytes. Rechecks the complete model SHA-256 in sequence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

from image_weights import PART_SIZE, fetch_part, part_count, part_path

BASE_PATHS = {'image_weights.py': 'opt/wan-image/image_weights.py',
              'manifest.json': 'opt/wan-image/manifest.json',
              'image_entrypoint.sh': 'opt/wan-image/image_entrypoint.sh',
              'extra_model_paths.yaml': 'opt/comfyui-baked/extra_model_paths.yaml'}


def run(*args, input_text=None, capture=False):
    result = subprocess.run(args, input=input_text, text=True,
                            check=True, stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else None


def make_layer(archive, files):
    with tarfile.open(archive, 'w') as out:
        for source, arcname, mode in files:
            info = out.gettarinfo(str(source), arcname)
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            info.mtime = 0
            info.mode = mode
            with Path(source).open('rb') as payload:
                out.addfile(info, payload)


def build(root, work, ref, crane='crane'):
    if os.environ.get('GITHUB_ACTIONS') != 'true' or not os.environ.get('GITHUB_TOKEN'):
        raise RuntimeError('Remote GitHub Actions and its ephemeral package token are required')
    if not ref.startswith('ghcr.io/') or ':' not in ref:
        raise ValueError('Expected a tagged GHCR image reference')
    manifest = json.loads((root / 'manifest.json').read_text())
    models = manifest['models']
    work.mkdir(parents=True, exist_ok=True)
    run(crane, 'auth', 'login', 'ghcr.io', '-u', os.environ['GITHUB_ACTOR'],
        '--password-stdin', input_text=os.environ['GITHUB_TOKEN'])
    # Copy pinned base remotely (never pull its 5.63 GB to the controller).
    run(crane, 'copy', '--platform=linux/amd64', manifest['base_image'], ref)
    layer = work / 'layer.tar'
    files = [(root / filename, archive_path, 0o755 if filename.endswith('.sh') else 0o644)
             for filename, archive_path in BASE_PATHS.items()]
    try:
        make_layer(layer, files)
        run(crane, 'append', '--base', ref, '--new_layer', str(layer), '--new_tag', ref)
    finally:
        layer.unlink(missing_ok=True)
    parts_root = work / 'opt/wan-weight-parts'
    for model in models:
        digest = hashlib.sha256()
        for index in range(part_count(model)):
            result = fetch_part(model, index, parts_root)
            part = part_path(model, parts_root, index)
            expected = min(PART_SIZE, model['size'] - index * PART_SIZE)
            if part.stat().st_size != expected:
                raise ValueError('Incorrect model part size')
            with part.open('rb') as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(block)
            try:
                make_layer(layer, [(part, 'opt/wan-weight-parts/' + model['destination'] + f'.part{index:02d}', 0o644)])
                run(crane, 'append', '--base', ref, '--new_layer', str(layer), '--new_tag', ref)
            finally:
                layer.unlink(missing_ok=True)
                part.unlink(missing_ok=True)
            print('published verified-length part', model['destination'], index, result, flush=True)
        if digest.hexdigest() != model['sha256']:
            raise ValueError('Complete downloaded model did not match pinned SHA-256')
        print('full model SHA-256 verified:', model['destination'], flush=True)
    final = ref.rsplit(':', 1)[0] + ':v1'
    run(crane, 'mutate', ref, '--entrypoint=/opt/wan-image/image_entrypoint.sh',
        '--label=org.opencontainers.image.source=https://github.com/shea786/wan-fun-control-runpod-image',
        '--tag', final)
    config = json.loads(run(crane, 'config', final, capture=True))
    if config['config']['Entrypoint'] != ['/opt/wan-image/image_entrypoint.sh']:
        raise RuntimeError('Final image entrypoint readback did not match')
    image = json.loads(run(crane, 'manifest', final, capture=True))
    expected_layers = len(json.loads(run(crane, 'manifest', manifest['base_image'], capture=True))['layers']) + 1 + sum(part_count(m) for m in models)
    if len(image['layers']) != expected_layers:
        raise RuntimeError('Final image has the wrong number of layers')
    image_digest = run(crane, 'digest', final, capture=True)
    print('IMAGE_REFERENCE', final.rsplit(':', 1)[0] + '@' + image_digest, flush=True)
    print('IMAGE_COMPRESSED_BYTES', sum(x['size'] for x in image['layers']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    args = parser.parse_args()
    if os.environ.get('GITHUB_ACTIONS') != 'true':
        raise RuntimeError('Do not build or download weights on the controller')
    with tempfile.TemporaryDirectory(prefix='wan-image-build-') as temp:
        build(Path(__file__).parent, Path(temp), args.reference)


if __name__ == '__main__':
    main()
