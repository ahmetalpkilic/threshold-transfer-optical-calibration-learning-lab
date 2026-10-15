import unittest
from unittest.mock import patch
import numpy as np
import policy as l
import repeat_model as r

class PilotTests(unittest.TestCase):
    def test_every_branch_exactly_charges_budget(self):
        for name in l.METHODS[:-1]:
            for triggered in (False,True):
                pilot,fresh,m=l.allocation(name,triggered)
                self.assertEqual(160+pilot+fresh+40*m,640)
        self.assertEqual(l.allocation('conditional',False),(80,0,10))
        self.assertEqual(l.allocation('conditional',True),(80,80,8))

    def test_alarm_family_and_strict_boundary(self):
        self.assertFalse(l.alarm([0,.4],[.25,.4]))
        self.assertTrue(l.alarm([0,.4],[.251,.4]))
        self.assertTrue(l.alarm([0,.4],[0,.149]))
        self.assertFalse(l.alarm([0,.4],[0,.4]))
        for value in (-1,float('nan')):
            with self.assertRaises(ValueError):l.alarm([0,0],[0,0],value)
        with self.assertRaises(ValueError):l.alarm([0,0],[0,float('nan')])

    def test_unpurchased_observations_do_not_change_estimates(self):
        z=np.random.default_rng(9).normal(size=(40,12,2));old=np.array([.1,.4]);fresh=np.array([.2,.8])
        for triggered,m in [(False,10),(True,8)]:
            a=l.estimate_policy(z,old,fresh,'conditional',triggered)
            changed=z.copy();changed[:,m:]=np.nan
            np.testing.assert_array_equal(a,l.estimate_policy(changed,old,fresh,'conditional',triggered))
            used=fresh if triggered else old
            np.testing.assert_array_equal(a,r.estimate(z[:,:m],used[r.LABELS]))
        a=l.estimate_policy(z,old,fresh,'conditional',False)
        np.testing.assert_array_equal(a,l.estimate_policy(z,old,[np.nan,np.nan],'conditional',False))
        a=l.estimate_policy(z,old,fresh,'conditional',True)
        np.testing.assert_array_equal(a,l.estimate_policy(z,[np.nan,np.nan],fresh,'conditional',True))

    def test_forced_branch_matches_independent_policy_end_to_end(self):
        for flag,control in [(False,'pilot_never'),(True,'pilot_always')]:
            with patch.object(l,'alarm',return_value=flag):result=l.experiment(3,981)
            for row in result['rows']:
                self.assertEqual(row['alarm_rate'],float(flag))
                self.assertEqual(row['methods']['conditional'],row['methods'][control])

    def test_transfer_expectation_independent_ar_recursion(self):
        rng=np.random.default_rng(8);m=10;actual=.9;assumed=.4
        z=rng.normal(size=(15000,m,2))
        for j in range(1,m):z[:,j]=actual*z[:,j-1]+np.sqrt(1-actual**2)*z[:,j]
        q=r.residual_precision(m,assumed)
        observed=np.einsum('tki,kl,tlj->ij',z,q,z)/(len(z)*(m-1))
        expected=np.trace(q@r.correlation(m,actual))/(m-1)
        np.testing.assert_allclose(observed,expected*np.eye(2),atol=.012)

    def test_fresh_seed_determinism_and_accounting(self):
        a=l.experiment(3,917);self.assertEqual(a,l.experiment(3,917))
        for row in a['rows']:
            self.assertTrue(0<=row['alarm_rate']<=1)
            if row['current_rho']==l.OLD_RHO:self.assertIsNone(row['miss_rate'])
            else:self.assertAlmostEqual(row['miss_rate']+row['alarm_rate'],1)
            self.assertEqual(set(row['methods']),set(l.METHODS))
            for stats in row['methods'].values():self.assertTrue(np.isfinite(stats['rmse_ms']))
        with self.assertRaises(ValueError):l.experiment(1)

if __name__=='__main__':unittest.main()
