import unittest

from smollm_sampler import (
    SourceSpec,
    deterministic_page_indices,
    sample_source,
)


class FakeRows:
    def __init__(self, rows_by_config):
        self.rows_by_config = rows_by_config

    def __call__(self, config, offset, length):
        rows = self.rows_by_config[config]
        wrappers = [
            {"row_idx": index, "row": rows[index]}
            for index in range(offset, min(offset + length, len(rows)))
        ]
        return {"rows": wrappers, "num_rows_total": len(rows)}


class TestSmolLMSampler(unittest.TestCase):
    def test_page_order_is_deterministic_and_complete(self):
        first = deterministic_page_indices(
            total_rows=250,
            page_size=100,
            seed="same",
        )
        second = deterministic_page_indices(
            total_rows=250,
            page_size=100,
            seed="same",
        )
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), [0, 1, 2])

    def test_fineweb_quality_filter_and_token_budget(self):
        rows = [
            {
                "text": f"Document {index}",
                "id": f"id-{index}",
                "metadata": {
                    "int_score": 3 if index == 0 else 5,
                    "language": "en",
                    "language_score": 0.99,
                    "token_count": 10,
                },
            }
            for index in range(8)
        ]
        sample = sample_source(
            SourceSpec(
                config="fineweb-edu-dedup",
                output_name="fineweb.txt",
                token_budget=30,
                minimum_score=4,
            ),
            seed="test",
            page_size=8,
            fetcher=FakeRows({"fineweb-edu-dedup": rows}),
            progress=None,
        )

        self.assertGreaterEqual(sample["token_count"], 30)
        self.assertEqual(sample["document_count"], 3)
        self.assertEqual(sample["filtered_rows"], 1)

    def test_cosmopedia_duplicate_text_is_removed(self):
        rows = [
            {
                "text": "Repeated document",
                "token_length": 10,
                "audience": "general",
                "format": "story",
                "seed_data": "test",
            },
            {
                "text": " repeated   document ",
                "token_length": 10,
                "audience": "general",
                "format": "story",
                "seed_data": "test",
            },
            {
                "text": "Unique document",
                "token_length": 10,
                "audience": "general",
                "format": "textbook",
                "seed_data": "test",
            },
        ]
        sample = sample_source(
            SourceSpec(
                config="cosmopedia-v2",
                output_name="cosmopedia.txt",
                token_budget=20,
            ),
            seed="test",
            page_size=3,
            fetcher=FakeRows({"cosmopedia-v2": rows}),
            progress=None,
        )

        self.assertEqual(sample["document_count"], 2)
        self.assertEqual(sample["duplicate_rows"], 1)


if __name__ == "__main__":
    unittest.main()
