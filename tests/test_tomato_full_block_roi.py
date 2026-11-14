import json
from types import SimpleNamespace

import numpy as np

from scripts.tomato_full_block_roi import coordinates, export, interaction, sha


def test_right_truss_stays_in_view_when_left_hand_transports_away():
    rgb = np.zeros((400, 640, 3), np.uint8)
    rgb[230:245, 350:390] = (240, 0, 0)
    right = dict(label=1, score=.99, box=[360, 170, 440, 240])
    far_left = dict(label=2, score=.99, box=[20, 200, 130, 300])
    result = interaction(rgb, [right, far_left])
    assert result['box'][0] > 300
    assert result['box'][3] >= 245
    assert interaction(rgb, [far_left]) is None


def test_zoomed_path_preserves_local_geometry_and_reports_long_detection_gaps():
    detection = dict(frames=120, samples=[
        dict(frame=0, interaction=dict(box=[330, 180, 420, 260])),
        dict(frame=119, interaction=dict(box=[300, 200, 410, 280])),
    ])
    boxes, tight, nearest, checks = coordinates(detection)
    assert checks['all_boxes_contained']
    assert checks['minimum_side'] == 180
    assert checks['frames_over_15_from_anchor'] > 0
    assert not checks['independent_semantic_quality_accepted']
    assert checks['center_steps_over_3pct'] == checks['scale_steps_over_2pct'] == 0


def test_export_keeps_current_pixels_and_raw_bytes(tmp_path):
    import av
    from fractions import Fraction
    row = dict(category='normal', group='A', dataset='example', episode=0, fps=30,
               frames=12, video='normal/full-block-more/A/example/source.mp4')
    source = tmp_path / 'source'
    video = source / row['video']
    video.parent.mkdir(parents=True)
    with av.open(str(video), 'w') as handle:
        stream = handle.add_stream('libx264', rate=30)
        stream.width, stream.height = 640, 400
        stream.pix_fmt = 'yuv420p'
        for i in range(12):
            # A strongly changing scene distinguishes current RGB from old RGB.
            rgb = np.full((400, 640, 3), i * 18 + 20, np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            frame.pts, frame.time_base = i, Fraction(1, 30)
            for packet in stream.encode(frame):
                handle.mux(packet)
        for packet in stream.encode():
            handle.mux(packet)
    before = sha(video)
    detection = dict(frames=12, source_sha256=before, samples=[
        dict(frame=i, interaction=dict(box=[330, 180, 420, 260])) for i in (0, 11)
    ])
    receipt = export(row, source, tmp_path / 'out', detection)
    assert sha(video) == before
    assert receipt['decoded_frames'] == 12 and receipt['black_frames'] == 0
    assert receipt['checks']['same_current_frame_pixels']
    with av.open(str(tmp_path / 'out' / receipt['output_video'])) as handle:
        means = [f.to_ndarray(format='rgb24').mean() for f in handle.decode(video=0)]
    assert np.all(np.diff(means) > 10)
    assert not receipt['training_ready']
    assert not receipt['subtask_labels_generated']


def test_publication_copies_raw_bytes_without_links_and_refuses_conflicts(tmp_path, monkeypatch):
    import pytest
    import scripts.tomato_full_block_roi as module
    base = tmp_path / 'datasets'
    rel = 'normal/full-block-more/A/example'
    raw = base / rel
    (raw / 'meta').mkdir(parents=True)
    feature = dict(dtype='video', shape=[400, 640, 3], info={'video.width': 640, 'video.height': 400})
    info = dict(features={'observation.images.cam0': feature}, total_videos=3, total_episodes=1)
    (raw / 'meta/info.json').write_text(json.dumps(info))
    (raw / 'state.parquet').write_bytes(b'original state and action bytes')
    (raw / 'cam0.mp4').write_bytes(b'original global video bytes')
    row = dict(category='normal', group='A', dataset='example', relative_dataset=rel,
               episode=0, frames=12, fps=30, video=f'{rel}/cam0.mp4')
    monkeypatch.setattr(module, 'inventory', lambda _: [row])
    files = [dict(path=str(p.relative_to(base)), bytes=p.stat().st_size, sha256=sha(p))
             for p in raw.rglob('*') if p.is_file()]
    frozen = tmp_path / 'frozen.json'
    frozen.write_text(json.dumps(dict(files=files)))
    destination = base / 'normal/full-block-more/subtask-roi/A/example'
    video = destination / 'videos/chunk-000/observation.images.interaction/episode_000000.mp4'
    video.parent.mkdir(parents=True)
    video.write_bytes(b'verified current frame crop video')
    roi = destination / 'roi/chunk-000'
    roi.mkdir(parents=True)
    coords = roi / 'episode_000000.jsonl.gz'
    coords.write_bytes(b'verified coordinate sidecar')
    receipt = dict(source=row, source_sha256=sha(raw / 'cam0.mp4'), output_sha256=sha(video),
                   coordinates_sha256=sha(coords), frames=12, decoded_frames=12, black_frames=0,
                   checks=dict(all_boxes_contained=True, center_steps_over_3pct=0, scale_steps_over_2pct=0))
    (roi / 'episode_000000.receipt.json').write_text(json.dumps(receipt))
    (destination / 'visual_review.json').write_text(json.dumps(dict(
        episodes=[dict(episode=0, output_sha256=sha(video))])))
    args = SimpleNamespace(source=base, output=base, freeze=frozen, datasets=['example'])
    module.publish(args)
    assert sha(raw / 'meta/info.json') == next(f['sha256'] for f in files if f['path'].endswith('info.json'))
    assert sha(destination / 'state.parquet') == sha(raw / 'state.parquet')
    assert (destination / 'state.parquet').stat().st_ino != (raw / 'state.parquet').stat().st_ino
    assert json.loads((destination / 'meta/info.json').read_text())['total_videos'] == 4
    # A previous batch must remain in the index and keep its original bytes.
    old = base / 'normal/full-block-more/subtask-roi/B/previous_batch'
    old.mkdir(parents=True)
    (old / 'roi_manifest.json').write_text('{"previous_batch":true}\n')
    (old / 'state.parquet').write_bytes(b'previous batch state')
    old_digest = sha(old / 'roi_manifest.json')
    root_index = base / 'normal/full-block-more/subtask-roi/manifest.json'
    existing = json.loads(root_index.read_text())
    existing['packages'].append(dict(path=str(old), category='normal', group='B',
                                     episodes=4, frames=48, manifest_sha256=old_digest))
    root_index.write_text(json.dumps(existing))
    module.publish(args)  # Idempotent while preserving unrelated batches.
    merged = json.loads(root_index.read_text())
    assert merged['episodes'] == 5 and merged['frames'] == 60
    assert sha(old / 'roi_manifest.json') == old_digest
    assert (old / 'state.parquet').read_bytes() == b'previous batch state'
    (destination / 'state.parquet').write_bytes(b'user changed derived file')
    with pytest.raises(ValueError, match='Existing package file differs'):
        module.publish(args)
    assert (raw / 'state.parquet').read_bytes() == b'original state and action bytes'


def test_dataset_selection_rejects_missing_names_instead_of_processing_everything():
    import pytest
    from scripts.tomato_full_block_roi import select_datasets
    rows = [dict(dataset='old', episode=0), dict(dataset='new', episode=0)]
    assert select_datasets(rows, ['new']) == [rows[1]]
    with pytest.raises(ValueError, match='Requested datasets missing'):
        select_datasets(rows, ['typo'])
