import unittest
import numpy as np

from services.TFGPositioningService import build_tfg_mlp

class TestTFGPositioningService(unittest.TestCase):

    def test_mlp_training(self):

        rng = np.random.default_rng(42)
        rssi = rng.uniform(-90, -40, size=(120, 8))
        presence = rng.binomial(1, 0.85, size=(120, 8))

        rssi[presence == 0] = -100.0

        X = np.column_stack((rssi, presence))
        y = np.column_stack(((rssi[:, 0] + 100)/10, (rssi[:, 1] + 100)/10))

        X_train = X[:96]
        X_test = X[96:]

        for use_mask in [False, True]:

            model = build_tfg_mlp(use_mask = use_mask)
            model.fit(X_train, y[:96])

            predictions = model.predict(X_test)

            self.assertEqual(predictions.shape, (24, 2))
            self.assertTrue(np.isfinite(predictions).all())

            expected_features = 16 if use_mask else 8
            
            transformed = model.named_steps["preprocessor"].transform(X_test)
            self.assertEqual(transformed.shape[1], expected_features)

            if use_mask:
                np.testing.assert_array_equal(transformed[:, 8:], X_test[:, 8:])
            
            print(
                f"\nMLP use_mask = {use_mask}"
                f"\nEntrenamiento: {len(X_train)} capturas"
                f"\nEvaluación: {len(X_test)} capturas"
                f"\nPredicciones: {predictions.shape}"
            )

if __name__ == "__main__":
    unittest.main()