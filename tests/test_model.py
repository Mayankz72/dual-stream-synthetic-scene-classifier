import numpy as np
import torch
from PIL import Image

from scene_classifier.edge_transform import EdgeTransform
from scene_classifier.model import ResBlock, SEBlock, ShapeNet


def test_resblock_preserves_shape():
    block = ResBlock(16)
    x = torch.randn(2, 16, 32, 32)
    assert block(x).shape == x.shape


def test_seblock_preserves_shape():
    block = SEBlock(16)
    x = torch.randn(2, 16, 32, 32)
    assert block(x).shape == x.shape


def test_shapenet_output_shape():
    model = ShapeNet(in_channels=3)
    x = torch.randn(4, 3, 192, 192)
    out = model(x)
    assert out.shape == (4, 1)


def test_shapenet_resolution_agnostic():
    model = ShapeNet(in_channels=3)
    for size in (64, 128, 192):
        x = torch.randn(1, 3, size, size)
        assert model(x).shape == (1, 1)


def test_edge_transform_output_is_three_channel_image():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (100, 100, 3), dtype=np.uint8))
    out = EdgeTransform()(img)
    assert out.size == img.size
    assert np.array(out).shape[-1] == 3
