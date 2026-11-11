from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model, pi0, pi0_config
from openpi.policies.fr3_sharpa_crop_policy import Fr3SharpaCropInputs
from openpi.subtasks.rgb100 import IMAGE_KEYS
from openpi.training.config import LeRobotFr3SharpaCropDataConfig


def raw(valid=True):
    return {'observation/state':np.arange(116,dtype=np.float32),
        'observation/image':np.zeros((224,224,3),dtype=np.uint8),
        'observation/left_wrist_image':np.full((224,224,3),20,dtype=np.uint8),
        'observation/right_wrist_image':np.full((224,224,3),40,dtype=np.uint8),
        'observation/interaction_image':np.full((224,224,3),200,dtype=np.uint8),
        'observation/interaction_valid':np.bool_(valid),
        'actions':np.arange(50*58,dtype=np.float32).reshape(50,58),'prompt':'pickup'}


def test_opt_in_model_spec_keeps_default_three_and_declares_four():
    default=pi0_config.Pi0Config(pi05=True,action_dim=58,max_token_len=256)
    four=pi0_config.Pi0Config(pi05=True,action_dim=58,max_token_len=256,image_keys=IMAGE_KEYS)
    assert tuple(default.inputs_spec()[0].images)==model.IMAGE_KEYS
    obs,actions=four.inputs_spec(batch_size=2)
    assert tuple(obs.images)==tuple(obs.image_masks)==IMAGE_KEYS
    assert obs.tokenized_prompt.shape==(2,256) and actions.shape==(2,50,58)
    with pytest.raises(ValueError):pi0_config.Pi0Config(image_keys=('base','base'))


def test_crop_adapter_preserves_physical_mapping_and_mask():
    data=raw();out=Fr3SharpaCropInputs(model.ModelType.PI05)(data)
    expected=np.concatenate([data['observation/state'][:7],data['observation/state'][28:50],
        data['observation/state'][14:21],data['observation/state'][72:94]])
    np.testing.assert_array_equal(out['state'],expected)
    assert tuple(out['image'])==IMAGE_KEYS and out['image_mask']['interaction_0_rgb']
    out=Fr3SharpaCropInputs(model.ModelType.PI05)(raw(False))
    assert not out['image_mask']['interaction_0_rgb']
    del data['observation/interaction_valid']
    with pytest.raises(KeyError):Fr3SharpaCropInputs(model.ModelType.PI05)(data)


def test_repack_requires_matching_model_and_routes_actual_fourth_pixels():
    config=LeRobotFr3SharpaCropDataConfig()
    with pytest.raises(ValueError):config.create_vector_transforms(pi0_config.Pi0Config(action_dim=58))
    c=pi0_config.Pi0Config(pi05=True,action_dim=58,image_keys=IMAGE_KEYS)
    repack,vectors=config.create_vector_transforms(c)
    r=raw();mapping=repack.inputs[0].structure
    source={name:r[key] for key,name in mapping.items()}
    got=vectors.inputs[0](repack.inputs[0](source))
    np.testing.assert_array_equal(got['image']['interaction_0_rgb'],r['observation/interaction_image'])


def test_preprocessing_keeps_crop_and_mask_but_default_stays_three():
    c=pi0_config.Pi0Config(pi05=True,action_dim=58,image_keys=IMAGE_KEYS)
    obs=c.fake_obs()
    obs=obs.replace(images={**obs.images,'interaction_0_rgb':jnp.full((1,224,224,3),.5)},
        image_masks={**obs.image_masks,'interaction_0_rgb':jnp.array([False])})
    got=model.preprocess_observation(None,obs,image_keys=c.image_keys)
    np.testing.assert_array_equal(got.images['interaction_0_rgb'],obs.images['interaction_0_rgb'])
    assert not got.image_masks['interaction_0_rgb'][0]
    assert tuple(model.preprocess_observation(None,obs).images)==model.IMAGE_KEYS


def test_interaction_crop_augmentation_never_geometrically_removes_edge_evidence(monkeypatch):
    c=pi0_config.Pi0Config(action_dim=58,image_keys=IMAGE_KEYS)
    called=[]
    # Full image gets geometry; crop, like wrist, keeps geometry intact.
    original=model.augmax.RandomCrop
    def spy(*args,**kwargs):called.append(args);return original(*args,**kwargs)
    monkeypatch.setattr(model.augmax,'RandomCrop',spy)
    model.preprocess_observation(jax.random.key(0),c.fake_obs(),train=True,image_keys=IMAGE_KEYS)
    assert len(called)==1


def test_real_prefix_assembly_has1024_visual_plus256_text_and_fourth_view_gradients():
    c=pi0_config.Pi0Config(pi05=True,action_dim=58,max_token_len=256,image_keys=IMAGE_KEYS)
    obs=c.fake_obs()
    # Small differentiable encoder tests the actual prefix assembly, not pretrained quality.
    def image_encoder(x,train=False):
        return jnp.broadcast_to(x.mean((1,2,3))[:,None,None],(len(x),256,8)),None
    def llm(tokens,method=None):return jnp.zeros((*tokens.shape,8))
    probe=SimpleNamespace(image_keys=IMAGE_KEYS,PaliGemma=SimpleNamespace(img=image_encoder,llm=llm))
    tokens,mask,ar=pi0.Pi0.embed_prefix(probe,obs)
    assert tokens.shape==(1,1280,8) and mask.shape==(1,1280) and ar.shape==(1280,)
    def value(crop):
        o=obs.replace(images={**obs.images,'interaction_0_rgb':crop})
        return pi0.Pi0.embed_prefix(probe,o)[0][:,768:1024].sum()
    grad=jax.grad(value)(obs.images['interaction_0_rgb'])
    assert np.asarray(grad).sum()>0


def test_torch_preprocessing_uses_same_explicit_four_keys():
    import torch
    from openpi.models_pytorch.preprocessing_pytorch import preprocess_observation_pytorch
    image=torch.zeros((1,224,224,3))
    obs=model.Observation(images=dict.fromkeys(IMAGE_KEYS,image),
        image_masks=dict.fromkeys(IMAGE_KEYS,torch.ones(1,dtype=torch.bool)),state=torch.zeros((1,58)))
    got=preprocess_observation_pytorch(obs,image_keys=IMAGE_KEYS)
    assert tuple(got.images)==IMAGE_KEYS


def test_crop_adapter_vector_statistics_keep_arm_delta_and_hand_absolute():
    from examples.fr3_wuji.compute_norm_stats import transform_vectors, supports_vector_statistics
    from openpi import transforms
    c=pi0_config.Pi0Config(pi05=True,action_dim=58,image_keys=IMAGE_KEYS)
    data=LeRobotFr3SharpaCropDataConfig(extra_delta_transform=True)
    repack,vectors=data.create_vector_transforms(c)
    assert all(supports_vector_statistics(t) for t in vectors.inputs)
    state=np.arange(5*116,dtype=np.float32).reshape(5,116)
    actions=np.arange(5*58,dtype=np.float32).reshape(5,58)
    rows=np.array([1,3])
    out=transform_vectors(state,actions,rows,3,transforms.compose([*repack.inputs,*vectors.inputs]),
        camera_keys=('observation.images.cam0','observation.images.cam1','observation.images.cam2','observation.images.interaction'))
    q=np.concatenate([state[rows,:7],state[rows,28:50],state[rows,14:21],state[rows,72:94]],axis=-1)
    a=actions[np.minimum(rows[:,None]+np.arange(3),4)]
    expected=np.concatenate([a[:,:,:7],a[:,:,14:36],a[:,:,7:14],a[:,:,36:58]],axis=-1)
    expected[:,:,:7]-=q[:,None,:7];expected[:,:,29:36]-=q[:,None,29:36]
    np.testing.assert_array_equal(out['state'],q)
    np.testing.assert_array_equal(out['actions'],expected)
