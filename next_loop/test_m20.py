import unittest
import numpy as np
from next_loop.m20 import transform, panel


class TailChecks(unittest.TestCase):
    def test_boundaries_identity_monotonicity_and_no_mutation(self):
        p=np.array([300.,399.,400.,425.,450.,451.,600.]); saved=p.copy()
        np.testing.assert_array_equal(transform(p,0,0),p)
        for row in panel():
            q=transform(p,row['a'],row['b'])
            self.assertTrue(np.all(np.diff(q)>0))
            self.assertEqual(q[2],400); self.assertEqual(q[4],450)
        np.testing.assert_array_equal(p,saved)

    def test_nine_unique_recipes(self):
        self.assertEqual(len(panel()),9)
        self.assertEqual(len({(r['a'],r['b']) for r in panel()}),9)


if __name__=='__main__':unittest.main()
