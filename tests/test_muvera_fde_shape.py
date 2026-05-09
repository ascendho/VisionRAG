import unittest

import numpy as np
from fastembed.postprocess.muvera import Muvera

from src.vector_store import (
    COLPALI_VECTOR_DIM,
    MUVERA_DIM_PROJ,
    MUVERA_K_SIM,
    MUVERA_R_REPS,
    get_muvera_fde_dimension,
)


class MuveraFdeShapeTests(unittest.TestCase):
    def test_project_muvera_dimension_matches_fastembed(self):
        muvera = Muvera(
            dim=COLPALI_VECTOR_DIM,
            k_sim=MUVERA_K_SIM,
            dim_proj=MUVERA_DIM_PROJ,
            r_reps=MUVERA_R_REPS,
            random_seed=42,
        )

        self.assertEqual(get_muvera_fde_dimension(), 30720)
        self.assertEqual(muvera.embedding_size, get_muvera_fde_dimension())

    def test_muvera_process_outputs_single_fde_vector(self):
        dim = 8
        k_sim = 2
        dim_proj = 4
        r_reps = 3
        muvera = Muvera(dim=dim, k_sim=k_sim, dim_proj=dim_proj, r_reps=r_reps, random_seed=42)
        vectors = np.arange(5 * dim, dtype=np.float32).reshape(5, dim) / 10.0
        expected_dimension = get_muvera_fde_dimension(k_sim=k_sim, dim_proj=dim_proj, r_reps=r_reps)

        document_fde = muvera.process_document(vectors)
        query_fde = muvera.process_query(vectors[:2])

        self.assertEqual(document_fde.ndim, 1)
        self.assertEqual(query_fde.ndim, 1)
        self.assertEqual(document_fde.shape, (expected_dimension,))
        self.assertEqual(query_fde.shape, (expected_dimension,))


if __name__ == "__main__":
    unittest.main()