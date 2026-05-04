import unittest

from backend.api.routes.rag import _merge_result_groups, _select_results_with_threshold_fallback


def _make_result(page_number: int, score: float, document_id: str = "doc-1"):
    return {
        "document_id": document_id,
        "document_name": "sample.pdf",
        "page_number": page_number,
        "image_path": f"/tmp/{document_id}_{page_number}.png",
        "score": score,
    }


class EvidenceSelectionTests(unittest.TestCase):
    def test_supplements_near_threshold_results_up_to_selection_limit(self):
        selection = _select_results_with_threshold_fallback(
            [
                _make_result(11, 0.63),
                _make_result(9, 0.599),
                _make_result(12, 0.58),
            ],
            min_score=0.6,
            selection_limit=3,
        )

        self.assertEqual([item["page_number"] for item in selection["selected_results"]], [11, 9, 12])
        self.assertEqual([item["fallback_below_threshold"] for item in selection["selected_results"]], [False, True, True])
        self.assertTrue(selection["fallback_used"])
        self.assertEqual(selection["fallback_count"], 2)
        self.assertEqual(selection["fallback_best_score"], 0.6)

    def test_promotes_best_available_when_all_candidates_are_far_below_threshold(self):
        selection = _select_results_with_threshold_fallback(
            [
                _make_result(4, 0.51),
                _make_result(6, 0.49),
            ],
            min_score=0.6,
            selection_limit=3,
            fallback_limit=1,
        )

        self.assertEqual(len(selection["selected_results"]), 1)
        self.assertEqual(selection["selected_results"][0]["page_number"], 4)
        self.assertTrue(selection["selected_results"][0]["fallback_below_threshold"])
        self.assertEqual(selection["selected_results"][0]["fallback_reason"], "best_available_below_threshold<0.60")

    def test_merge_result_groups_keeps_highest_score_for_duplicate_page(self):
        merged = _merge_result_groups(
            [
                [_make_result(8, 0.59)],
                [_make_result(8, 0.61)],
            ],
            ["first query", "second query"],
            limit=3,
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["page_number"], 8)
        self.assertEqual(merged[0]["score"], 0.61)
        self.assertEqual(merged[0]["matched_sub_queries"], ["first query", "second query"])


if __name__ == "__main__":
    unittest.main()