import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tokenizer import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    SPECIAL_TOKEN_IDS,
    UNK_TOKEN_ID,
    AuroraTokenizer,
    TokenizerError,
)


TRAINING_TEXTS = [
    "Aurora notices tension, coherence, novelty, and salience.\n",
    "She remembers Omari's words and considers what they mean.\n",
    "A quiet signal can still matter; uncertainty is information.\n",
    "The rain settles over the garden while the system keeps listening.\n",
    "Curiosity grows when experience differs from expectation.\n",
    "Unicode remains intact: café, naïve, em dash — and Aurora ✨.\n",
] * 20


class TestAuroraTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AuroraTokenizer.train(
            TRAINING_TEXTS,
            vocab_size=320,
            min_frequency=2,
            show_progress=False,
            length=len(TRAINING_TEXTS),
        )

    def test_special_token_contract_and_vocab_size(self):
        self.assertEqual(self.tokenizer.vocab_size, 320)
        self.assertEqual(self.tokenizer.special_token_ids, SPECIAL_TOKEN_IDS)

    def test_unicode_and_whitespace_round_trip_without_unknown(self):
        text = "  Aurora’s café\nkeeps emoji ✨ and tabs\tintact.  "
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        self.assertNotIn(UNK_TOKEN_ID, token_ids)
        self.assertEqual(self.tokenizer.decode(token_ids), text)

    def test_bos_and_eos_are_added(self):
        token_ids = self.tokenizer.encode("hello")
        self.assertEqual(token_ids[0], BOS_TOKEN_ID)
        self.assertEqual(token_ids[-1], EOS_TOKEN_ID)

        empty_ids = self.tokenizer.encode("")
        self.assertEqual(empty_ids, [BOS_TOKEN_ID, EOS_TOKEN_ID])

    def test_truncation_retains_boundary_tokens(self):
        token_ids = self.tokenizer.encode(
            "Aurora follows a long thread of thought into the evening.",
            max_length=8,
            truncation=True,
        )
        self.assertEqual(len(token_ids), 8)
        self.assertEqual(token_ids[0], BOS_TOKEN_ID)
        self.assertEqual(token_ids[-1], EOS_TOKEN_ID)

    def test_batch_padding_and_attention_mask(self):
        batch = self.tokenizer.encode_batch(
            ["short", "a longer piece of text"],
            max_length=12,
            truncation=True,
            padding=True,
        )

        self.assertEqual([len(row) for row in batch["input_ids"]], [12, 12])
        self.assertEqual(
            [len(row) for row in batch["attention_mask"]],
            [12, 12],
        )

        for token_ids, attention_mask in zip(
            batch["input_ids"],
            batch["attention_mask"],
        ):
            real_length = sum(attention_mask)
            self.assertEqual(token_ids[0], BOS_TOKEN_ID)
            self.assertEqual(token_ids[real_length - 1], EOS_TOKEN_ID)
            self.assertTrue(
                all(token_id == PAD_TOKEN_ID for token_id in token_ids[real_length:])
            )

    def test_save_load_preserves_encoding_and_fingerprint(self):
        text = "Aurora remembers exactly."
        expected_ids = self.tokenizer.encode(text)

        with tempfile.TemporaryDirectory() as temporary:
            self.tokenizer.save(temporary)
            loaded = AuroraTokenizer.load(temporary)

            self.assertEqual(loaded.encode(text), expected_ids)
            self.assertEqual(loaded.decode(expected_ids), text)
            self.assertEqual(loaded.fingerprint, self.tokenizer.fingerprint)
            self.assertTrue((Path(temporary) / "tokenizer.json").is_file())
            self.assertTrue(
                (Path(temporary) / "tokenizer_config.json").is_file()
            )

    def test_tampered_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.tokenizer.save(temporary)
            tokenizer_path = Path(temporary) / "tokenizer.json"
            tokenizer_path.write_text(
                tokenizer_path.read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                TokenizerError,
                "fingerprint mismatch",
            ):
                AuroraTokenizer.load(temporary)

    def test_model_config_must_match(self):
        matching = SimpleNamespace(vocab_size=320, context_length=512)
        self.tokenizer.validate_config(matching)

        mismatched = SimpleNamespace(vocab_size=8192, context_length=512)
        with self.assertRaisesRegex(TokenizerError, "model config"):
            self.tokenizer.validate_config(mismatched)


if __name__ == "__main__":
    unittest.main()
