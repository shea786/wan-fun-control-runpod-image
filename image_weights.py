"""Remote image builder: fetch chunked, pinned Wan weights and reassemble on a Pod.

Only image builds on a remote builder may fetch; runtime assembly requires a
RunPod Pod. Running this on the local controller is deliberately refused.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import quote
from urllib.request import Request, urlopen

PART_SIZE = 2_000_000_000  # safer on GitHub's 14 GB hosted runner; < GHCR 10 GB/layer
PARTS_ROOT = Path('/opt/wan-weight-parts')
MODELS_ROOT = Path('/workspace/wan-models')
MANIFEST = Path('/opt/wan-image/manifest.json')


def part_count(model, part_size=PART_SIZE):
    return (model['size'] + part_size - 1) // part_size


def safe_target(root, destination):
    destination = Path(destination)
    if destination.is_absolute() or not destination.parts or '..' in destination.parts:
        raise ValueError('Unsafe model path')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True)
    target = root / destination
    if not target.resolve(strict=False).is_relative_to(root):
        raise ValueError('Model path escapes root')
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def part_path(model, parts_root, part):
    return safe_target(parts_root, model['destination'] + f'.part{part:02d}')


def model_url(model):
    return 'https://huggingface.co/' + model['repo'] + '/resolve/' + model['revision'] + '/' + quote(model['path'], safe='/')


def fetch_part(model, part, parts_root=PARTS_ROOT, opener=urlopen, part_size=PART_SIZE):
    if part < 0 or part >= part_count(model, part_size):
        raise ValueError('Part index out of range')
    start = part * part_size
    end = min(model['size'], start + part_size) - 1
    size = end - start + 1
    target = part_path(model, parts_root, part)
    if target.is_file() and target.stat().st_size == size:
        return 'already-present'
    request = Request(model_url(model), headers={'Range': f'bytes={start}-{end}'})
    temp = None
    try:
        with opener(request, timeout=120) as response:
            content_range = response.headers.get('Content-Range', '')
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range)
            if response.status != 206 or not match or tuple(map(int, match.groups())) != (start, end, model['size']):
                raise ValueError('Remote source ignored or altered requested byte range')
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.part-', delete=False) as output:
                temp = Path(output.name)
                count = 0
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    count += len(chunk)
                    if count > size:
                        raise ValueError('Part exceeded expected byte range')
                    output.write(chunk)
        if count != size:
            raise ValueError('Part shorter than expected byte range')
        temp.replace(target)
        return 'downloaded'
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def stream_parts(model, parts_root, destination=None, part_size=PART_SIZE):
    digest = hashlib.sha256()
    total = 0
    for index in range(part_count(model, part_size)):
        part = part_path(model, parts_root, index)
        expected = min(part_size, model['size'] - index * part_size)
        if part.stat().st_size != expected:
            raise ValueError('Incorrect part size')
        with part.open('rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(chunk)
                total += len(chunk)
                if destination is not None:
                    destination.write(chunk)
    if total != model['size'] or digest.hexdigest() != model['sha256']:
        raise ValueError('Assembled model differs from pinned size or SHA-256')


def assemble(model, parts_root=PARTS_ROOT, models_root=MODELS_ROOT, part_size=PART_SIZE):
    target = safe_target(models_root, model['destination'])
    if target.is_file() and target.stat().st_size == model['size']:
        digest = hashlib.sha256()
        with target.open('rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() == model['sha256']:
            return 'already-present'
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.model-', delete=False) as output:
            temp = Path(output.name)
            stream_parts(model, parts_root, output, part_size)
        temp.replace(target)
        return 'assembled'
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('fetch-part', 'verify', 'assemble'))
    parser.add_argument('model', type=int, nargs='?')
    parser.add_argument('part', type=int, nargs='?')
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--parts-root', type=Path, default=PARTS_ROOT)
    parser.add_argument('--models-root', type=Path, default=MODELS_ROOT)
    args = parser.parse_args(argv)
    if args.action == 'assemble':
        if not os.environ.get('RUNPOD_POD_ID'):
            raise RuntimeError('Assembly is permitted only on a RunPod Pod')
    elif os.environ.get('RUNPOD_IMAGE_BUILD') != '1':
        raise RuntimeError('Fetching/verification is permitted only in the remote image build')
    models = json.loads(args.manifest.read_text(encoding='utf-8'))['models']
    if args.action == 'fetch-part':
        if args.model is None or args.part is None:
            parser.error('fetch-part needs a model and part index')
        print(fetch_part(models[args.model], args.part, args.parts_root), flush=True)
    elif args.action == 'verify':
        for model in models:
            stream_parts(model, args.parts_root)
            print('verified', model['destination'], flush=True)
    else:
        for model in models:
            print(assemble(model, args.parts_root, args.models_root), model['destination'], flush=True)


if __name__ == '__main__':
    main()
