import cv2
import numpy as np
import pytest
from scripts.tomato_causal_roi import CausalBoxTrack


def texture():
    return np.random.default_rng(42).integers(0,256,(100,120),dtype=np.uint8)


def test_follows_translation_without_later_annotations():
    image=texture();tracker=CausalBoxTrack(image,[20,20,70,70],0)
    moved=cv2.warpAffine(image,np.float32([[1,0,2],[0,1,1]]),(120,100))
    result=tracker.update(moved,1)
    assert result['valid']
    np.testing.assert_allclose(result['box'],[22,21,72,71],atol=.8)
    with pytest.raises(ValueError,match='Consecutive'):
        tracker.update(moved,3)


def test_lost_is_invalid_and_never_reuses_old_box():
    image=texture();tracker=CausalBoxTrack(image,[20,20,70,70],0)
    result=tracker.update(np.zeros_like(image),1)
    assert not result['valid'] and result['box'] is None
    result=tracker.update(image,2)
    assert not result['valid'] and result['box'] is None


def test_blank_seed_and_age_limit_require_explicit_reseed():
    empty=np.zeros((100,120),np.uint8);tracker=CausalBoxTrack(empty,[20,20,70,70],0)
    assert not tracker.update(empty,1)['valid']
    image=texture();tracker=CausalBoxTrack(image,[20,20,70,70],0,max_age=1)
    assert tracker.update(image,1)['valid']
    result=tracker.update(image,2)
    assert not result['valid'] and result['reason']=='track_age_limit'


def test_small_back_and_forth_rotation_does_not_inflate_box():
    image=texture();tracker=CausalBoxTrack(image,[30,25,70,65],0)
    for i,angle in enumerate([1,0]*15,1):
        moved=cv2.warpAffine(image,cv2.getRotationMatrix2D((50,45),angle,1),(120,100))
        result=tracker.update(moved,i)
        assert result['valid']
        b=np.array(result['box'])
        assert np.prod(b[2:]-b[:2]) < 1.2*40*40
