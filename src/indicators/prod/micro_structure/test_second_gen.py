import unittest

import numpy as np

# Assuming the functions are in the same directory or sys.path is configured
from second_gen import amihud_lambda, hasbrouck_lambda, kyle_lambda


class TestSecondGenMicroStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up dummy candle data for testing."""
        cls.window = 5
        n_candles = 30
        timestamps = np.arange(1678886400000, 1678886400000 + n_candles * 60000, 60000)
        # Simple price movement: mostly increasing, then decreasing slightly
        closes = np.concatenate(
            [
                np.linspace(100, 105, 10),
                np.linspace(105, 104.5, 5),
                np.linspace(104.5, 108, 10),
                np.linspace(108, 107.5, 5),
            ]
        )
        opens = closes - 0.1
        highs = closes + 0.2
        lows = closes - 0.3
        # Volumes: Include some variations, avoid zero initially for simplicity
        volumes = np.array(
            [
                10,
                12,
                15,
                13,
                11,
                10,
                14,
                16,
                18,
                20,
                22,
                25,
                23,
                21,
                19,
                17,
                15,
                10,
                5,
                8,
                9,
                11,
                13,
                16,
                19,
                22,
                25,
                28,
                30,
                26,
            ],
            dtype=float,
        )

        cls.candles = np.column_stack([timestamps, opens, highs, lows, closes, volumes])

        # Create a version with zero volume to test edge cases
        cls.candles_zero_vol = cls.candles.copy()
        cls.candles_zero_vol[cls.window + 2, 5] = (
            0  # Set a volume to zero after the first window
        )

    def test_kyle_lambda(self):
        """Test the kyle_lambda function."""
        # Test sequential=True
        res_seq = kyle_lambda(self.candles, window=self.window, sequential=True)
        self.assertIsInstance(res_seq, np.ndarray)
        self.assertEqual(res_seq.shape, (len(self.candles),))
        # First window-1 values should be 0 due to min_periods=window and nan_to_num
        np.testing.assert_array_equal(res_seq[: self.window - 1], 0)
        self.assertFalse(np.isnan(res_seq).any())
        self.assertFalse(np.isinf(res_seq).any())

        # Test sequential=False
        res_last = kyle_lambda(self.candles, window=self.window, sequential=False)
        self.assertIsInstance(res_last, np.float64)
        np.testing.assert_almost_equal(res_last, res_seq[-1])

        # Test with zero volume
        res_zero_vol = kyle_lambda(
            self.candles_zero_vol, window=self.window, sequential=True
        )
        self.assertFalse(np.isnan(res_zero_vol).any())
        self.assertFalse(np.isinf(res_zero_vol).any())
        # Expect 0 where division by zero occurred (due to volume=0)
        # The exact index depends on rolling window calculation
        # Let's check if there are more zeros than just the initial ones
        self.assertTrue(np.sum(res_zero_vol[self.window - 1 :] == 0) > 0)

    def test_amihud_lambda(self):
        """Test the amihud_lambda function."""
        # Test sequential=True
        res_seq = amihud_lambda(self.candles, window=self.window, sequential=True)
        self.assertIsInstance(res_seq, np.ndarray)
        self.assertEqual(res_seq.shape, (len(self.candles),))
        np.testing.assert_array_equal(res_seq[: self.window - 1], 0)
        self.assertFalse(np.isnan(res_seq).any())
        self.assertFalse(np.isinf(res_seq).any())

        # Test sequential=False
        res_last = amihud_lambda(self.candles, window=self.window, sequential=False)
        self.assertIsInstance(res_last, np.float64)
        np.testing.assert_almost_equal(res_last, res_seq[-1])

        # Test with zero volume (affects dollar volume)
        res_zero_vol = amihud_lambda(
            self.candles_zero_vol, window=self.window, sequential=True
        )
        self.assertFalse(np.isnan(res_zero_vol).any())
        self.assertFalse(np.isinf(res_zero_vol).any())
        self.assertTrue(np.sum(res_zero_vol[self.window - 1 :] == 0) > 0)

    def test_hasbrouck_lambda(self):
        """Test the hasbrouck_lambda function."""
        # Test sequential=True
        res_seq = hasbrouck_lambda(self.candles, window=self.window, sequential=True)
        self.assertIsInstance(res_seq, np.ndarray)
        self.assertEqual(res_seq.shape, (len(self.candles),))
        np.testing.assert_array_equal(res_seq[: self.window - 1], 0)
        self.assertFalse(np.isnan(res_seq).any())
        self.assertFalse(np.isinf(res_seq).any())

        # Test sequential=False
        res_last = hasbrouck_lambda(self.candles, window=self.window, sequential=False)
        self.assertIsInstance(res_last, np.float64)
        np.testing.assert_almost_equal(res_last, res_seq[-1])

        # Test with zero volume (affects dollar volume)
        res_zero_vol = hasbrouck_lambda(
            self.candles_zero_vol, window=self.window, sequential=True
        )
        self.assertFalse(np.isnan(res_zero_vol).any())
        self.assertFalse(np.isinf(res_zero_vol).any())
        self.assertTrue(np.sum(res_zero_vol[self.window - 1 :] == 0) > 0)


if __name__ == "__main__":
    unittest.main()
