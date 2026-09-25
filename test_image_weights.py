import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from image_weights import assemble, fetch_part, main, part_count, stream_parts


class RangeResponse(io.BytesIO):
    def __init__(self, payload, start, end):
        super().__init__(payload[start:end + 1])
        self.status = 206
        self.headers = {'Content-Range': f'bytes {start}-{end}/{len(payload)}'}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class ImageWeightsTests(unittest.TestCase):
    def setUp(self):
        self.data = b'fake model safetensors payload!'
        self.model = {'repo': 'Test/Model', 'revision': 'a' * 40,
                      'path': 'split_files/diffusion_models/model.safetensors',
                      'destination': 'diffusion_models/model.safetensors',
                      'size': len(self.data), 'sha256': hashlib.sha256(self.data).hexdigest()}
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parts = Path(self.temp.name) / 'parts'
        self.models = Path(self.temp.name) / 'models'

    def opener(self, request, timeout):
        self.assertEqual(timeout, 120)
        self.assertIn('/resolve/' + 'a' * 40 + '/', request.full_url)
        range_value = request.get_header('Range')
        self.assertIsNotNone(range_value)
        start, end = (int(n) for n in range_value.removeprefix('bytes=').split('-'))
        return RangeResponse(self.data, start, end)

    def test_fetch_verify_assemble_and_reuse(self):
        self.assertEqual(part_count(self.model, 11), 3)
        for part in range(3):
            self.assertEqual(fetch_part(self.model, part, self.parts, self.opener, 11), 'downloaded')
        self.assertEqual(fetch_part(self.model, 0, self.parts, self.opener, 11), 'already-present')
        stream_parts(self.model, self.parts, part_size=11)
        self.assertEqual(assemble(self.model, self.parts, self.models, 11), 'assembled')
        self.assertEqual((self.models / self.model['destination']).read_bytes(), self.data)
        self.assertEqual(assemble(self.model, self.parts, self.models, 11), 'already-present')

    def test_rejects_missing_or_corrupted_part_and_does_not_publish(self):
        for part in range(3):
            fetch_part(self.model, part, self.parts, self.opener, 11)
        corrupt = self.parts / (self.model['destination'] + '.part01')
        corrupt.write_bytes(b'!' * 11)
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            assemble(self.model, self.parts, self.models, 11)
        self.assertFalse((self.models / self.model['destination']).exists())

    def test_rejects_ignored_range_and_oversize(self):
        def ignores_range(*_args, **_kwargs):
            response = RangeResponse(self.data, 0, len(self.data) - 1)
            response.status = 200
            response.headers = {}
            return response
        with self.assertRaisesRegex(ValueError, 'byte range'):
            fetch_part(self.model, 0, self.parts, ignores_range, 11)
        with self.assertRaisesRegex(ValueError, 'out of range'):
            fetch_part(self.model, 3, self.parts, self.opener, 11)

    def test_refuses_local_build_or_assembly(self):
        manifest = Path(self.temp.name) / 'manifest.json'
        manifest.write_text(json.dumps({'models': [self.model]}))
        with patch.dict(os.environ, {'RUNPOD_IMAGE_BUILD': '', 'RUNPOD_POD_ID': ''}):
            for action in ('fetch-part', 'verify', 'assemble'):
                with self.assertRaisesRegex(RuntimeError, 'only'):
                    main([action, '--manifest', str(manifest)])

    def test_rejects_escape_destination_and_symlink(self):
        for destination in ('/absolute', '../outside', 'diffusion_models/../../outside'):
            self.model['destination'] = destination
            with self.assertRaises(ValueError):
                fetch_part(self.model, 0, self.parts, self.opener, 11)
        self.model['destination'] = 'diffusion_models/model.safetensors'
        self.parts.mkdir(exist_ok=True)
        (self.parts / 'diffusion_models').symlink_to(Path(self.temp.name) / 'outside', target_is_directory=True)
        with self.assertRaises(ValueError):
            fetch_part(self.model, 0, self.parts, self.opener, 11)


if __name__ == '__main__':
    unittest.main()
