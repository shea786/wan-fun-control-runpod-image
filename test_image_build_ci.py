import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from image_build_ci import BASE_PATHS, build, make_layer


class ImageBuildTests(unittest.TestCase):
    def test_layer_paths_and_executable_entrypoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            script = root / 'image_entrypoint.sh'
            script.write_text('#!/bin/bash\nexit 0\n')
            target = root / 'layer.tar'
            make_layer(target, [(script, 'opt/wan-image/image_entrypoint.sh', 0o755)])
            with tarfile.open(target) as archive:
                item = archive.getmember('opt/wan-image/image_entrypoint.sh')
                self.assertEqual(item.mode, 0o755)
                self.assertEqual(archive.extractfile(item).read(), script.read_bytes())

    def test_only_expected_code_and_metadata_enter_image(self):
        self.assertEqual(set(BASE_PATHS), {'image_weights.py', 'manifest.json', 'image_entrypoint.sh', 'extra_model_paths.yaml'})
        manifest = json.loads(Path(__file__).with_name('manifest.json').read_text())
        self.assertEqual(set(manifest), {'base_image', 'models'})
        self.assertEqual(len(manifest['models']), 4)
        self.assertNotIn('q22x8xwdz6', json.dumps(manifest))

    def test_refuses_build_without_hosted_runner_and_ephemeral_token(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict(os.environ, {'GITHUB_ACTIONS': '', 'GITHUB_TOKEN': ''}):
                with self.assertRaisesRegex(RuntimeError, 'GitHub Actions'):
                    build(Path(temp), Path(temp), 'ghcr.io/example/image:tag')


if __name__ == '__main__':
    unittest.main()
