import unittest
from unittest.mock import patch
import numpy as np
import lab
from characterization import ar_sequence, reml_score, fit_rho

class ThresholdTests(unittest.TestCase):
    def test_rank_and_ties_without_interpolation(self):
        h,k=lab.rank_threshold(np.arange(999),.05)
        self.assertEqual(k,950);self.assertEqual(h,949)
        self.assertLessEqual((1000-k)/1000,.05)
        self.assertEqual(lab.rank_threshold(np.ones(99))[0],1)
        for scores,alpha in [([], .05),([1,2],.05),([np.nan]*99,.05),(np.ones(99),0)]:
            with self.assertRaises(ValueError):lab.rank_threshold(scores,alpha)

    def test_exact_rank_count_by_leave_one_out(self):
        # Independent combinatorial check of the marginal rank argument.
        scores=np.arange(20);alarms=0
        for i in range(20):
            h,k=lab.rank_threshold(np.delete(scores,i),.2)
            alarms+=scores[i]>h
        self.assertEqual(alarms,4)

    def test_affine_reml_score_identity_and_estimator(self):
        z=ar_sequence(np.random.default_rng(42).normal(size=(40,2)),.4,np.eye(2),np.zeros(2))
        a=np.array([[1.5,.4],[-.2,.8]]); transformed=z@a.T+[17,-8]
        expected=2*(len(z)-1)*np.log(abs(np.linalg.det(a)))
        for rho in (-.8,0,.4,.9):
            self.assertAlmostEqual(reml_score(transformed,rho)-reml_score(z,rho),expected,places=9)
        self.assertAlmostEqual(fit_rho(transformed),fit_rho(z),places=6)

    def test_changed_covariance_is_positive_definite_and_different(self):
        cov=lab.nuisance_covariance()
        self.assertTrue(np.all(np.linalg.eigvalsh(cov)>0))
        self.assertFalse(np.allclose(cov,lab.c.covariances(4.)))
        def corr(x):return x[:,0,1]/np.sqrt(x[:,0,0]*x[:,1,1])
        self.assertFalse(np.allclose(corr(cov),corr(lab.c.covariances(4.))))

    def test_tuning_does_not_depend_on_final_measurements(self):
        # Same root and train count; changing number of test images cannot retune.
        a=lab.experiment(19,2,123);b=lab.experiment(19,3,123)
        self.assertEqual(a['threshold'],b['threshold'])
        self.assertEqual(a['training_scores'],b['training_scores'])
        self.assertLess(a['max_paired_score_difference'],2e-6)
        self.assertEqual(a,lab.experiment(19,2,123))
        for row in a['rows']:
            self.assertTrue(all(np.isfinite(s['rmse_ms']) for s in row['methods'].values()))

if __name__=='__main__':unittest.main()
