"""Test the utils module of TensorVision."""


def test_set_dirs():
    """Test if setting plugins works."""
    hype_file = "examples/cifar10_minimal.json"
    with open(hype_file, 'r') as f:
        import json
        hypes = json.load(f)

    from tensorvision.utils import set_dirs
    import tensorflow as tf
    flags = tf.app.flags
    flags.DEFINE_string('name', 'debug',
                        'Append a name Tag to run.')

    tf.app.flags.DEFINE_boolean(
        'save', True, (''))

    set_dirs(hypes, hype_file)


def test_cfg():
    """Test if one can call the cfg as expected."""
    from tensorvision.utils import cfg
    cfg.plugin_dir


def test_load_plugins():
    """Test if loading plugins works."""
    from tensorvision.utils import load_plugins
    load_plugins()


def test_overlay_segmentation():
    """Test overlay_segmentation."""
    from tensorvision.utils import overlay_segmentation
    import scipy.ndimage
    import numpy as np
    seg_image = 'tensorvision/tests/Crocodylus-johnsoni-3-seg.png'
    original_image = 'tensorvision/tests/Crocodylus-johnsoni-3.jpg'
    target_path = 'tensorvision/tests/croco-overlay-new.png'
    correct_path = 'tensorvision/tests/croco-overlay.png'
    input_image = scipy.misc.imread(original_image)
    segmentation = scipy.misc.imread(seg_image)
    color_changes = {0: (0, 255, 0, 127),
                     'default': (0, 0, 0, 0)}
    res = overlay_segmentation(input_image, segmentation, color_changes)
    scipy.misc.imsave(target_path, res)
    img1 = scipy.ndimage.imread(correct_path)
    img2 = scipy.ndimage.imread(target_path)
    assert np.array_equal(img1, img2)


def test_soft_overlay_segmentation():
    """Test soft_overlay_segmentation."""
    from tensorvision.utils import soft_overlay_segmentation
    import scipy.ndimage
    import numpy as np
    seg_image = 'tensorvision/tests/Crocodylus-johnsoni-3-seg-prob.png'
    original_image = 'tensorvision/tests/Crocodylus-johnsoni-3.jpg'
    target_path = 'tensorvision/tests/croco-overlay-new-soft.png'
    correct_path = 'tensorvision/tests/croco-overlay-soft.png'
    input_image = scipy.misc.imread(original_image)
    segmentation = scipy.misc.imread(seg_image) / 255.0
    res = soft_overlay_segmentation(input_image, segmentation)
    scipy.misc.imsave(target_path, res)
    img1 = scipy.ndimage.imread(correct_path)
    img2 = scipy.ndimage.imread(target_path)
    assert np.array_equal(img1, img2)


def test_load_segmentation_mask():
    """Test load_segmentation_mask."""
    from tensorvision.utils import load_segmentation_mask
    import json
    import scipy.misc
    import numpy
    import os
    hypes_path = os.path.abspath('tensorvision/tests/croco.json')
    seg_image = os.path.abspath('tensorvision/tests/croco-mask.png')
    with open(hypes_path) as f:
        hypes = json.load(f)
    gt = load_segmentation_mask(hypes, seg_image)
    # scipy.misc.imsave("tensorvision/tests/test-generated-mask.png", gt)
    seg_image = 'tensorvision/tests/Crocodylus-johnsoni-3-mask.png'
    true_gt = (scipy.misc.imread(seg_image) / 255.).astype(int)
    numpy.testing.assert_array_equal(true_gt, gt)
    assert gt.shape == (480, 640)
