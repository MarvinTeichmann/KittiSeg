"""Test the utils module of TensorVision."""


def test_get_confusion_matrix():
    """Test if get_confusion_matrix works."""
    from scipy.misc import imread
    from tensorvision.analyze import get_confusion_matrix
    gt = imread('tensorvision/tests/Crocodylus-johnsoni-3-mask.png', mode='L')
    seg = imread('tensorvision/tests/Crocodylus-johnsoni-3-seg.png', mode='L')
    n = get_confusion_matrix(gt, seg, [0, 255])
    assert n == {0: {0: 46832, 255: 1669}, 255: {0: 5347, 255: 253352}}
    assert seg.shape[0] * seg.shape[1] == sum(x
                                              for c in n.values()
                                              for x in c.values())


def test_get_accuracy():
    """Test if get_accuracy works."""
    from tensorvision.analyze import get_accuracy
    n = {0: {0: 10, 1: 2}, 1: {0: 5, 1: 83}}
    assert abs(get_accuracy(n) - 0.93) <= 0.0001


def test_get_mean_accuracy():
    """Test if get_mean_accuracy works."""
    from tensorvision.analyze import get_mean_accuracy
    n = {0: {0: 10, 1: 2}, 1: {0: 5, 1: 83}}
    assert abs(get_mean_accuracy(n) - 0.8882575757575758) <= 0.0001


def test_get_mean_iou():
    """Test if get_mean_iou works."""
    from tensorvision.analyze import get_mean_iou
    n = {0: {0: 10, 1: 2}, 1: {0: 5, 1: 83}}
    assert abs(get_mean_iou(n) - 0.7552287581699346) <= 0.0001


def test_get_frequency_weighted_iou():
    """Test if get_frequency_weighted_iou works."""
    from tensorvision.analyze import get_frequency_weighted_iou
    n = {0: {0: 10, 1: 2}, 1: {0: 5, 1: 83}}
    assert abs(get_frequency_weighted_iou(n) - 0.8821437908496732) <= 0.0001


def test_get_precision():
    """Test if get_precision works."""
    from tensorvision.analyze import get_precision
    n = {0: {0: 10, 1: 2}, 1: {0: 5, 1: 83}}
    assert abs(get_precision(n) - 0.9764705882352941) <= 0.0001


def test_get_recall():
    """Test if get_recall works."""
    from tensorvision.analyze import get_recall
    n = {0: {0: 10, 1: 2}, 1: {0: 5, 1: 83}}
    assert abs(get_recall(n) - 0.9431818181818182) <= 0.0001


def test_f_score():
    """Test if get_f_score works."""
    from tensorvision.analyze import get_f_score
    n = {0: {0: 10, 1: 2}, 1: {0: 5, 1: 83}}
    assert abs(get_f_score(n) - 0.959537572254) <= 0.0001
    assert abs(get_f_score(n, beta=0.5) - 0.969626168224) <= 0.0001


def test_merge_cms():
    """Test if merge_cms works."""
    from tensorvision.analyze import merge_cms
    cm1 = {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}
    cm2 = {0: {0: 5, 1: 6}, 1: {0: 7, 1: 8}}
    cmr = {0: {0: 6, 1: 8}, 1: {0: 10, 1: 12}}
    assert merge_cms(cm1, cm2) == cmr
