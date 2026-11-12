import numpy as np
import pytest
from scripts.prepare_tomato100_available_crops import image_plan,render
from scripts.materialize_tomato100_four_view import read_candidate_index
import gzip
import json


def row():
    return dict(source_frame=0,timestamp=0,available_at=0,valid=True,crop_box=[1,1,7,7],interaction_box=[2,2,6,6],source_size=[10,10])


def write(path,entry):
    with gzip.open(path,'wt') as stream:stream.write(json.dumps(entry)+'\n')


def test_loss_uses_current_full_pixels_and_retains_invalid_localization(tmp_path):
    entry=row();entry.update(valid=False,crop_box=None,interaction_box=None,reason='required_hand_lost')
    entry.update(image_plan(entry))
    assert entry['image_available'] and not entry['local_roi_valid']
    assert entry['input_mode']=='full_view_fallback'
    image=np.full((10,10,3),[20,80,170],np.uint8)
    np.testing.assert_array_equal(render(image,entry)[112,112],image[0,0])
    image[:]=[130,20,40]
    np.testing.assert_array_equal(render(image,entry)[112,112],image[0,0])
    p=tmp_path/'index.gz';write(p,entry)
    assert read_candidate_index(p,1)[0]['image_available']
    entry['image_available']=False;write(p,entry)
    with pytest.raises(ValueError,match='availability'):read_candidate_index(p,1)


def test_local_input_and_letterboxed_fallback_are_distinct():
    entry=row();entry.update(image_plan(entry))
    assert entry['input_mode']=='local_crop' and entry['local_roi_valid']
    image=np.zeros((10,10,3),np.uint8);image[1:7,1:7]=180
    assert render(image,entry).mean()==180
    entry.update(valid=False,crop_box=None,interaction_box=None,source_size=[20,10],reason='window_catching_up')
    entry.update(image_plan(entry))
    crop=render(np.full((10,20,3),180,np.uint8),entry)
    assert not crop[:56].any() and np.all(crop[56:168]==180) and not crop[168:].any()
