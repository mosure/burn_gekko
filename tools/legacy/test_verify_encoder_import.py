import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from verify_encoder_import import verify


class EncoderImportIntegrity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'src').mkdir()
        self.source = self.root / 'src/model.rs'
        self.source.write_text('original source')
        self.original = hashlib.sha256(self.source.read_bytes()).hexdigest()
        (self.root / 'UPSTREAM.json').write_text(json.dumps(dict(
            revision='pinned-revision', files=[dict(source='src/model.rs',sha256=self.original)])))

    def record_patch(self, revision='pinned-revision', original=None):
        self.source.write_text('reviewed adaptation')
        local = hashlib.sha256(self.source.read_bytes()).hexdigest()
        (self.root / 'LOCAL_MODIFICATIONS.toml').write_text(f'''
schema = 1
upstream_revision = "{revision}"
[[files]]
source = "src/model.rs"
upstream_sha256 = "{original or self.original}"
local_sha256 = "{local}"
changes = ["recorded source adaptation"]
verification = "independent source and behavior review"
''')

    def test_unmodified_import_is_verified_and_unrecorded_change_rejected(self):
        self.assertEqual(verify(self.root)['verbatim'],1)
        self.source.write_text('unrecorded change')
        with self.assertRaisesRegex(ValueError,'outside its recorded hash'):
            verify(self.root)

    def test_exact_recorded_patch_is_allowed_but_further_tampering_is_not(self):
        self.record_patch()
        self.assertEqual(verify(self.root)['adapted'],1)
        self.source.write_text('unreviewed further change')
        with self.assertRaisesRegex(ValueError,'outside its recorded hash'):
            verify(self.root)

    def test_patch_cannot_rebind_original_revision_or_source_hash(self):
        self.record_patch(revision='different')
        with self.assertRaisesRegex(ValueError,'different upstream revision'):
            verify(self.root)
        self.record_patch(original='0'*64)
        with self.assertRaisesRegex(ValueError,'wrong original hash'):
            verify(self.root)


if __name__ == '__main__':
    unittest.main()
