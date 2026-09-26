import unittest

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data import ContrastiveDataset, make_splits
from src.evaluation import embedding_similarity_samples, metric_dict
from src.losses import nt_xent_loss
from src.models import Classifier, CompactEncoder, SimCLR, make_linear_probe


class TinyImages(Dataset):
    def __init__(self):
        self.images = [Image.fromarray(np.full((28, 28), i * 20, dtype=np.uint8)) for i in range(10)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], index  # the wrapper must discard this label


class PipelineTests(unittest.TestCase):
    def test_model_dimensions_and_shared_encoder_architecture(self):
        x = torch.randn(4, 1, 28, 28)
        supervised_encoder = CompactEncoder(128)
        contrastive_encoder = CompactEncoder(128)
        self.assertEqual(supervised_encoder(x).shape, (4, 128))
        self.assertEqual(Classifier(supervised_encoder)(x).shape, (4, 10))
        self.assertEqual(SimCLR(contrastive_encoder)(x).shape, (4, 64))
        self.assertEqual(str(supervised_encoder), str(contrastive_encoder))

    def test_two_augmented_views_have_expected_shape_and_no_label(self):
        dataset = ContrastiveDataset(TinyImages(), np.arange(10))
        sample = dataset[3]
        self.assertIsInstance(sample, tuple)
        self.assertEqual(len(sample), 2)
        self.assertEqual(sample[0].shape, (1, 28, 28))
        self.assertEqual(sample[1].shape, (1, 28, 28))
        self.assertTrue(torch.isfinite(sample[0]).all())
        self.assertFalse(torch.equal(sample[0], sample[1]))

    def test_nt_xent_is_finite_and_backpropagates(self):
        z1 = torch.randn(8, 16, requires_grad=True)
        z2 = torch.randn(8, 16, requires_grad=True)
        loss = nt_xent_loss(z1, z2, temperature=0.5)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(z1.grad)
        self.assertGreater(z1.grad.abs().sum().item(), 0)

    def test_splits_are_disjoint_and_stratified(self):
        train_labels = np.tile(np.arange(10), 100)
        test_labels = np.tile(np.arange(10), 20)
        split = make_splits(train_labels, test_labels, 600, 200, 100, seed=7)
        self.assertEqual(np.intersect1d(split.train, split.val).size, 0)
        self.assertEqual(len(np.unique(split.test)), 100)
        self.assertTrue(np.all(np.bincount(train_labels[split.train], minlength=10) == 60))

    def test_linear_probe_freezes_encoder(self):
        encoder = CompactEncoder()
        probe = make_linear_probe(encoder)
        self.assertTrue(all(not p.requires_grad for p in probe.encoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in probe.classifier.parameters()))

    def test_multiclass_macro_metrics(self):
        truth = np.array([0, 0, 1, 1, 2, 2])
        prediction = np.array([0, 1, 1, 1, 2, 0])
        metrics = metric_dict(truth, prediction)
        self.assertAlmostEqual(metrics["accuracy"], 4 / 6)
        self.assertAlmostEqual(metrics["recall_macro"], (0.5 + 1.0 + 0.5) / 3)
        self.assertEqual(np.asarray(metrics["confusion_matrix"]).shape, (10, 10))

    def test_similarity_diagnostic_matches_identical_views(self):
        from torch.utils.data import DataLoader, TensorDataset
        views = torch.eye(4).reshape(4, 1, 2, 2)
        diagnostic = embedding_similarity_samples(
            torch.nn.Flatten(), DataLoader(TensorDataset(views, views), batch_size=4),
            torch.device("cpu"), max_batches=1
        )
        self.assertTrue(np.allclose(diagnostic["positive"], 1.0))
        self.assertTrue(np.isfinite(diagnostic["negative"]).all())

    def test_update_budget_expands_small_label_training(self):
        from src.training import epochs_for_updates, labelled_batch_size
        self.assertEqual(labelled_batch_size(120, 128), 16)
        self.assertEqual(epochs_for_updates(3, loader_length=8, min_updates=60), 8)

    def test_training_interfaces_cannot_receive_test_loader_for_selection(self):
        # Model-selection functions expose train/validation only; test evaluation is separate.
        import inspect
        from src.training import train_classifier, train_contrastive
        self.assertNotIn("test_loader", inspect.signature(train_classifier).parameters)
        self.assertNotIn("test_loader", inspect.signature(train_contrastive).parameters)


if __name__ == "__main__":
    unittest.main()
