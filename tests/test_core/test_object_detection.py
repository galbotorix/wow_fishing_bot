from pathlib import Path

import cv2
import pytest

import core.object_detection as object_detection_module
from core.object_detection import ObjectDetection
from core.screenshot_frame import ScreenshotFrame
from support.app_error_handling import TemplateNotFoundError
from support.constants import assets_dir_name, templates_dir_name

ASSETS_DIR = Path(object_detection_module.__file__).parent.parent / assets_dir_name / templates_dir_name

KNOWN_BOBBER_POSITION = (1018, 176)  # confirmed via find_bobber() on the real fixture screenshot (Nov 2026 capture)

# The original screenshots (older bobber/session) are kept for the
# brightness splash detector, which only works with that older look.
LEGACY_DIR = Path(object_detection_module.__file__).parent.parent / "tests" / "fixtures" / "legacy"
LEGACY_BOBBER_POSITION = (901, 425)
LEGACY_SPLASH_BOBBER_POSITION = (940, 375)
FIXTURES_DIR = LEGACY_DIR.parent


def load_frame(filename: str, directory: Path = ASSETS_DIR, origin: tuple[int, int] = (0, 0)) -> ScreenshotFrame:
    """Loads one of the real fixture screenshots as a ScreenshotFrame (origin = its top-left on screen)."""
    image = cv2.imread(str(directory / filename))
    assert image is not None, f"Failed to load fixture image: {filename}"
    return ScreenshotFrame(image=image, timestamp=0.0, origin=origin)


# The legacy screenshots are stored as 200x200 crops around the bobber;
# the origin puts them back at their original on-screen position.
LEGACY_NO_SPLASH = ("screenshot_with_bobber_no_splash_crop.jpg", (801, 325))
LEGACY_WITH_SPLASH = ("screenshot_with_bobber_with_splash_crop.jpg", (840, 275))


def test_templates_are_loaded_on_init(objects):
    """
    The bobber template must load successfully from the real asset
    file, since it's present in assets/templates/. (Splash detection no
    longer uses a template -- see detect_splash tests below.)
    """
    object_detection = objects['object_detection']

    assert object_detection.bobber_template_gray is not None


def test_load_template_gray_returns_none_for_missing_file():
    """
    A missing template file should log a warning and return None,
    instead of raising, so the bot can still start without real assets.
    """
    missing_path = ASSETS_DIR / "does_not_exist.jpg"
    assert not missing_path.exists()

    result = ObjectDetection._ObjectDetection__load_template_gray(missing_path)
    assert result is None


def test_load_template_gray_raises_for_corrupt_file(tmp_path):
    """
    An existing but unreadable/corrupt image file should raise
    TemplateNotFoundError instead of silently returning None.
    """
    corrupt_file = tmp_path / "corrupt.jpg"
    corrupt_file.write_bytes(b"not a real image")

    with pytest.raises(TemplateNotFoundError):
        ObjectDetection._ObjectDetection__load_template_gray(corrupt_file)


def test_find_bobber_returns_none_when_no_bobber_present(objects):
    """
    find_bobber() must return None on a screenshot where no bobber is
    visible at all.
    """
    object_detection = objects['object_detection']
    frame = load_frame("screenshot_no_bobber.jpg")

    assert object_detection.find_bobber(frame) is None


def test_find_bobber_finds_bobber_when_present(objects):
    """
    find_bobber() must locate the bobber on a screenshot where it's
    floating on the water (no splash yet), returning a position close
    to the known real location.
    """
    object_detection = objects['object_detection']
    frame = load_frame("screenshot_with_bobber_no_splash.jpg")

    position = object_detection.find_bobber(frame)

    assert position is not None
    x, y = position
    expected_x, expected_y = KNOWN_BOBBER_POSITION
    assert abs(x - expected_x) <= 5
    assert abs(y - expected_y) <= 5


def test_find_bobber_respects_frame_origin_offset(objects):
    """
    find_bobber() must add frame.origin to the locally-matched position,
    so a non-zero origin (e.g. from a cropped capture) is handled correctly.
    """
    object_detection = objects['object_detection']
    image = cv2.imread(str(ASSETS_DIR / "screenshot_with_bobber_no_splash.jpg"))
    offset_origin = (100, 50)
    frame = ScreenshotFrame(image=image, timestamp=0.0, origin=offset_origin)

    position = object_detection.find_bobber(frame)

    assert position is not None
    x, y = position
    expected_x = KNOWN_BOBBER_POSITION[0] + offset_origin[0]
    expected_y = KNOWN_BOBBER_POSITION[1] + offset_origin[1]
    assert abs(x - expected_x) <= 5
    assert abs(y - expected_y) <= 5


def test_find_bobber_returns_none_when_template_missing(objects, monkeypatch):
    """
    If the bobber template failed to load (None), find_bobber() must
    return None instead of raising.
    """
    object_detection = objects['object_detection']
    monkeypatch.setattr(object_detection, "bobber_template_gray", None)

    frame = load_frame("screenshot_with_bobber_no_splash.jpg")
    assert object_detection.find_bobber(frame) is None


def test_find_bobber_returns_none_when_frame_smaller_than_template(objects):
    """
    A frame smaller than the template in either dimension must be
    handled gracefully, returning None instead of raising from
    cv2.matchTemplate.
    """
    object_detection = objects['object_detection']
    tiny_image = cv2.imread(str(ASSETS_DIR / "bobber.jpg"))[:5, :5]
    frame = ScreenshotFrame(image=tiny_image, timestamp=0.0, origin=(0, 0))

    assert object_detection.find_bobber(frame) is None


def test_detect_splash_false_when_no_splash_present(objects):
    """
    detect_splash() must return False for a frame where the bobber is
    floating but hasn't splashed yet.
    """
    object_detection = objects['object_detection']
    frame = load_frame(LEGACY_NO_SPLASH[0], LEGACY_DIR, LEGACY_NO_SPLASH[1])

    assert object_detection.detect_splash(frame, LEGACY_BOBBER_POSITION, mode="brightness") is False


def test_detect_splash_true_when_splash_present(objects):
    """
    detect_splash() must return True for a frame where the bobber has
    just splashed.
    """
    object_detection = objects['object_detection']
    frame = load_frame(LEGACY_WITH_SPLASH[0], LEGACY_DIR, LEGACY_WITH_SPLASH[1])

    assert object_detection.detect_splash(frame, LEGACY_SPLASH_BOBBER_POSITION, mode="brightness") is True


def test_detect_splash_returns_false_when_region_out_of_frame(objects):
    """
    If the splash region computed around bobber_position doesn't
    overlap the frame at all, detect_splash() must return False
    instead of raising.
    """
    object_detection = objects['object_detection']
    frame = load_frame("screenshot_with_bobber_with_splash.jpg")
    out_of_bounds_position = (-10_000, -10_000)

    assert object_detection.detect_splash(frame, out_of_bounds_position, mode="brightness") is False


# ----- Multi-scale matching --------------------------------------------------

def test_find_bobber_still_finds_bobber_when_template_is_scaled_down(objects, monkeypatch):
    """
    If the real bobber appears SMALLER on screen than the stored
    template (e.g. the player zoomed out since the template was
    captured), find_bobber() must still locate it by trying smaller
    scaled-down versions of the template.
    """
    object_detection = objects['object_detection']

    # Shrink the real fixture screenshot itself to 70% size, simulating
    # the bobber appearing smaller (more "zoomed out") than the template.
    original = cv2.imread(str(ASSETS_DIR / "screenshot_with_bobber_no_splash.jpg"))
    shrunk = cv2.resize(original, (int(original.shape[1] * 0.7), int(original.shape[0] * 0.7)))
    frame = ScreenshotFrame(image=shrunk, timestamp=0.0, origin=(0, 0))

    position = object_detection.find_bobber(frame)

    assert position is not None
    expected_x = int(KNOWN_BOBBER_POSITION[0] * 0.7)
    expected_y = int(KNOWN_BOBBER_POSITION[1] * 0.7)
    assert abs(position[0] - expected_x) <= 10
    assert abs(position[1] - expected_y) <= 10


def test_multi_scale_match_prefers_the_best_scoring_scale(objects):
    """
    __multi_scale_match must return the highest score found across all
    configured scales, not just whatever the first scale happens to produce.
    """
    object_detection = objects['object_detection']
    frame = load_frame("screenshot_with_bobber_no_splash.jpg")
    frame_gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)

    best_score, best_loc, best_shape = ObjectDetection._ObjectDetection__multi_scale_match(
        frame_gray, object_detection.bobber_template_gray, [0.5, 1.0, 1.5]
    )

    # Manually compute the score at scale=1.0 (no resizing) and confirm
    # the multi-scale result is at least as good (it found the real
    # bobber, cropped at native scale, so 1.0 should already be near-perfect).
    import cv2 as cv2_local
    result = cv2_local.matchTemplate(frame_gray, object_detection.bobber_template_gray, cv2_local.TM_CCOEFF_NORMED)
    _, single_scale_score, _, _ = cv2_local.minMaxLoc(result)

    assert best_score >= single_scale_score - 1e-6
    assert best_score > 0.9  # this fixture was cropped from this exact screenshot


def test_top_n_match_locations_returns_distinct_sorted_candidates(objects):
    """
    __top_n_match_locations must return results sorted best-first, with
    no two candidates closer together than min_distance -- otherwise
    near-duplicate matches at adjacent pixels/scales would dominate the
    top-N instead of genuinely distinct candidate locations.
    """
    object_detection = objects['object_detection']
    frame = load_frame("screenshot_with_bobber_no_splash.jpg")
    frame_gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)

    candidates = ObjectDetection._ObjectDetection__top_n_match_locations(
        frame_gray, object_detection.bobber_template_gray, [0.5, 1.0, 1.5], n=3, min_distance=50
    )

    assert len(candidates) >= 1
    assert len(candidates) <= 3
    scores = [c[0] for c in candidates]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > 0.9  # the real bobber match, same fixture as above

    locations = [c[1] for c in candidates]
    for i in range(len(locations)):
        for j in range(i + 1, len(locations)):
            dist = ((locations[i][0] - locations[j][0]) ** 2 + (locations[i][1] - locations[j][1]) ** 2) ** 0.5
            assert dist >= 50


# ----- Color-blob fallback -----------------------------------------------------

def test_find_bobber_uses_color_fallback_when_template_score_is_low(objects, monkeypatch):
    """
    If the template match score is below threshold, find_bobber() must
    still locate the bobber via the color-blob fallback (seeded by the
    template's best guess), rather than giving up immediately.
    """
    object_detection = objects['object_detection']
    frame = load_frame("screenshot_with_bobber_no_splash.jpg")

    # Force the template match to "fail" (as if captured at a very
    # different zoom level), so only the color-blob fallback can succeed.
    monkeypatch.setattr(object_detection_module, "bobber_match_threshold", 1.01)

    position = object_detection.find_bobber(frame)

    assert position is not None
    expected_x, expected_y = KNOWN_BOBBER_POSITION
    # The color-blob fallback is less precise than a good template
    # match, so allow a more generous tolerance.
    assert abs(position[0] - expected_x) <= 30
    assert abs(position[1] - expected_y) <= 30


def test_find_bobber_tries_next_candidate_when_best_scoring_seed_is_a_false_lead(objects, monkeypatch):
    """
    Regression test for a real missed detection: the single
    GLOBAL-best-scoring template location is not always the real
    bobber -- an unrelated location elsewhere can occasionally score
    almost identically (or even higher). find_bobber() must still find
    the bobber via a lower-ranked candidate rather than giving up after
    the top candidate's color-blob search fails.
    """
    object_detection = objects['object_detection']
    frame = load_frame("screenshot_with_bobber_no_splash.jpg")
    monkeypatch.setattr(object_detection_module, "bobber_match_threshold", 1.01)

    from support.constants import bobber_search_region_x_fraction, bobber_search_region_y_fraction
    full_height, full_width = frame.image.shape[:2]
    x_start, _ = ObjectDetection._ObjectDetection__fraction_bounds(full_width, bobber_search_region_x_fraction)
    y_start, _ = ObjectDetection._ObjectDetection__fraction_bounds(full_height, bobber_search_region_y_fraction)

    real_x, real_y = KNOWN_BOBBER_POSITION
    template_height, template_width = object_detection.bobber_template_gray.shape[:2]
    # Top-left location (relative to the cropped search region) that
    # would produce KNOWN_BOBBER_POSITION as its center.
    real_seed_local = (real_x - template_width // 2 - x_start, real_y - template_height // 2 - y_start)
    fake_seed_local = (10, 10)  # a location far from the real bobber, with no feather colors nearby

    fake_candidate = (0.90, fake_seed_local, (template_height, template_width))  # ranked first, but false
    real_candidate = (0.85, real_seed_local, (template_height, template_width))  # ranked second, genuine

    monkeypatch.setattr(
        ObjectDetection, "_ObjectDetection__top_n_match_locations",
        staticmethod(lambda *args, **kwargs: [fake_candidate, real_candidate])
    )

    position = object_detection.find_bobber(frame)

    assert position is not None
    assert abs(position[0] - real_x) <= 30
    assert abs(position[1] - real_y) <= 30


def test_find_bobber_returns_none_without_seed_when_template_missing(objects, monkeypatch):
    """
    The color-blob fallback requires a template-based seed location to
    search near it (scanning the whole central region without one is
    too noisy in practice). If no template is loaded at all, find_bobber()
    must return None rather than attempting an unseeded color scan.
    """
    object_detection = objects['object_detection']
    monkeypatch.setattr(object_detection, "bobber_template_gray", None)

    frame = load_frame("screenshot_with_bobber_no_splash.jpg")
    assert object_detection.find_bobber(frame) is None


# ----- Bobber motion confirmation --------------------------------------------

def test_is_bobber_candidate_static_true_for_identical_frames(objects):
    """
    If the area around the candidate position is pixel-identical between
    two frames, it must be flagged as static (a likely false positive on
    a background/UI element, since real water always has some ripple).
    """
    object_detection = objects['object_detection']
    image = cv2.imread(str(ASSETS_DIR / "screenshot_with_bobber_no_splash.jpg"))

    frame_a = ScreenshotFrame(image=image, timestamp=0.0, origin=(0, 0))
    frame_b = ScreenshotFrame(image=image.copy(), timestamp=0.1, origin=(0, 0))

    assert object_detection.is_bobber_candidate_static(frame_a, frame_b, KNOWN_BOBBER_POSITION) is True


def test_is_bobber_candidate_static_false_when_region_changes(objects):
    """
    If the area around the candidate position visibly changes between
    the two frames (simulated here by brightening it), it must NOT be
    flagged as static.
    """
    object_detection = objects['object_detection']
    image_a = cv2.imread(str(ASSETS_DIR / "screenshot_with_bobber_no_splash.jpg"))
    image_b = image_a.copy()

    x, y = KNOWN_BOBBER_POSITION
    # Brighten a patch around the candidate position to simulate motion/ripple.
    image_b[y - 20:y + 20, x - 20:x + 20] = cv2.add(image_b[y - 20:y + 20, x - 20:x + 20], 100)

    frame_a = ScreenshotFrame(image=image_a, timestamp=0.0, origin=(0, 0))
    frame_b = ScreenshotFrame(image=image_b, timestamp=0.1, origin=(0, 0))

    assert object_detection.is_bobber_candidate_static(frame_a, frame_b, KNOWN_BOBBER_POSITION) is False


def test_is_bobber_candidate_static_false_when_region_out_of_bounds(objects):
    """
    An inconclusive comparison (position outside the frame) must not be
    treated as "static" -- it shouldn't block a legitimate candidate
    just because the check itself couldn't run.
    """
    object_detection = objects['object_detection']
    frame = load_frame("screenshot_with_bobber_no_splash.jpg")
    out_of_bounds_position = (-10_000, -10_000)

    assert object_detection.is_bobber_candidate_static(frame, frame, out_of_bounds_position) is False


# ----- Bobber feather color confirmation --------------------------------------

def test_has_bobber_feather_colors_true_at_the_real_bobber_position(objects):
    """
    The real bobber position must have both the blue and red feather
    colors detected nearby.
    """
    object_detection = objects['object_detection']
    frame = load_frame("screenshot_with_bobber_no_splash.jpg")

    assert object_detection.has_bobber_feather_colors(frame, KNOWN_BOBBER_POSITION) is True


def test_has_bobber_feather_colors_false_on_plain_water(objects):
    """
    A position over plain water (no bobber at all) must not have the
    feather colors detected -- real water lacks the saturated blue/red
    hues entirely.
    """
    object_detection = objects['object_detection']
    frame = load_frame("screenshot_no_bobber.jpg")

    # Same coordinates as the (now patched-over) bobber position in the
    # synthetic no-bobber fixture.
    assert object_detection.has_bobber_feather_colors(frame, KNOWN_BOBBER_POSITION) is False


def test_has_bobber_feather_colors_true_when_region_out_of_bounds(objects):
    """
    An inconclusive check (position outside the frame) must not block a
    legitimate candidate just because the check itself couldn't run.
    """
    object_detection = objects['object_detection']
    frame = load_frame("screenshot_with_bobber_no_splash.jpg")
    out_of_bounds_position = (-10_000, -10_000)

    assert object_detection.has_bobber_feather_colors(frame, out_of_bounds_position) is True




# --- Motion-based splash detection -------------------------------------------

import numpy as np

MOTION_POSITION = (100, 100)


def make_water_frame(seed: int, splash: bool = False) -> ScreenshotFrame:
    """
    A synthetic 200x200 "water" frame: a fixed texture with a little
    random shimmer each frame (normal ripples). With splash=True, a
    burst of bright/dark blobs is drawn around the bobber, like a bite.
    """
    base = np.random.default_rng(0).integers(40, 90, (200, 200), dtype=np.uint8)
    shimmer = np.random.default_rng(seed).integers(-4, 5, (200, 200))
    gray = np.clip(base.astype(int) + shimmer, 0, 255).astype(np.uint8)
    if splash:
        rng = np.random.default_rng(1000 + seed)
        for _ in range(40):
            cx, cy = rng.integers(70, 130, 2)
            cv2.circle(gray, (int(cx), int(cy)), int(rng.integers(3, 8)), int(rng.integers(0, 255)), -1)
    return ScreenshotFrame(image=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), timestamp=0.0, origin=(0, 0))


def test_motion_splash_not_triggered_by_normal_water(objects):
    object_detection = objects['object_detection']
    object_detection.reset_splash_watch()

    results = [object_detection.detect_splash(make_water_frame(i), MOTION_POSITION, mode="motion")
               for i in range(20)]

    assert not any(results)


def test_motion_splash_detected_when_burst_of_motion_appears(objects):
    object_detection = objects['object_detection']
    object_detection.reset_splash_watch()

    for i in range(6):
        assert object_detection.detect_splash(make_water_frame(i), MOTION_POSITION, mode="motion") is False

    assert object_detection.detect_splash(make_water_frame(99, splash=True), MOTION_POSITION, mode="motion") is True


def test_motion_splash_never_fires_while_still_learning_baseline(objects):
    """Even a big change can't trigger before the baseline frames are collected."""
    object_detection = objects['object_detection']
    object_detection.reset_splash_watch()

    assert object_detection.detect_splash(make_water_frame(0), MOTION_POSITION, mode="motion") is False
    assert object_detection.detect_splash(make_water_frame(1, splash=True), MOTION_POSITION, mode="motion") is False


def test_reset_splash_watch_forgets_previous_cast(objects):
    object_detection = objects['object_detection']
    for i in range(6):
        object_detection.detect_splash(make_water_frame(i), MOTION_POSITION, mode="motion")

    object_detection.reset_splash_watch()

    # Straight after a reset the first frame only seeds the detector.
    assert object_detection.detect_splash(make_water_frame(99, splash=True), MOTION_POSITION, mode="motion") is False


def test_motion_watch_area_scales_with_bobber_size(objects):
    object_detection = objects['object_detection']
    assert object_detection.motion_radius_for_bobber_height(None) == 40      # unknown -> default
    assert object_detection.motion_radius_for_bobber_height(57) == 40        # close bobber
    assert object_detection.motion_radius_for_bobber_height(39) == 27        # far bobber
    assert object_detection.motion_radius_for_bobber_height(5) == 18         # clamped
    assert object_detection.motion_radius_for_bobber_height(500) == 60       # clamped


def test_either_mode_fires_on_bright_splash_fixture(objects):
    object_detection = objects['object_detection']
    object_detection.reset_splash_watch()
    frame = load_frame(LEGACY_WITH_SPLASH[0], LEGACY_DIR, LEGACY_WITH_SPLASH[1])

    assert object_detection.detect_splash(frame, LEGACY_SPLASH_BOBBER_POSITION, mode="either") is True


def test_unknown_splash_mode_raises(objects):
    object_detection = objects['object_detection']
    frame = make_water_frame(0)

    with pytest.raises(ValueError):
        object_detection.detect_splash(frame, MOTION_POSITION, mode="nope")


def load_real_bite_sequence():
    """
    Real capture (Nov 2026): the area around the cork bobber, one frame
    every 0.2s, from just after it lands until shortly after a bite.
    The bite starts at video frame 68 (t=6.7s).
    """
    data = np.load(FIXTURES_DIR / "real_bite_sequence.npz")
    origin = tuple(int(v) for v in data["origin"])
    frames = [ScreenshotFrame(image=cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), timestamp=0.0, origin=origin)
              for g in data["frames"]]
    return frames, [int(n) for n in data["frame_numbers"]]


REAL_BITE_FIRST_FRAME = 68
REAL_BITE_BOBBER_POSITION = (1017, 180)


@pytest.mark.parametrize("step", [1, 2])  # every 0.2s and every 0.4s
def test_motion_splash_detects_real_bite_without_false_alarms(objects, step):
    """
    Replays the real capture at the bot's cadence, from every possible
    starting offset: no detection may happen before the bite, and the
    bite itself must always be caught within a second.
    """
    object_detection = objects['object_detection']
    frames, numbers = load_real_bite_sequence()

    for offset in range(step):
        object_detection.reset_splash_watch()
        detected_at = None
        for frame, number in list(zip(frames, numbers))[offset::step]:
            if object_detection.detect_splash(frame, REAL_BITE_BOBBER_POSITION, mode="motion"):
                detected_at = number
                break
        assert detected_at is not None, f"missed the bite (offset {offset})"
        assert REAL_BITE_FIRST_FRAME <= detected_at <= REAL_BITE_FIRST_FRAME + 10, \
            f"detected at frame {detected_at} (offset {offset})"


def test_brightness_splash_false_alarms_on_new_cork_bobber(objects):
    """
    Documents why "motion" is the default: the new cork bobber is bright
    enough by itself to trip the brightness detector with no bite at all.
    """
    object_detection = objects['object_detection']
    frame = load_frame("screenshot_with_bobber_no_splash.jpg")

    assert object_detection.detect_splash(frame, KNOWN_BOBBER_POSITION, mode="brightness") is True


def load_sequence(name):
    data = np.load(FIXTURES_DIR / name)
    origin = tuple(int(v) for v in data["origin"])
    frames = [ScreenshotFrame(image=cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), timestamp=0.0, origin=origin)
              for g in data["frames"]]
    return frames, [float(t) for t in data["times"]]


@pytest.mark.parametrize("step", [1, 2, 3])  # every 0.1s, 0.2s, 0.3s
def test_motion_splash_detects_real_far_bite(objects, step):
    """
    Second real capture (Nov 2026): a FAR bobber (~45px tall instead of
    ~57-62px). Its bite (t=31.3s) is a weaker signal and was missed with a
    fixed-size watch area; the area now scales with the bobber's size.
    """
    object_detection = objects['object_detection']
    frames, times = load_sequence("real_far_bite_sequence.npz")
    position = (1072, 153)

    for offset in range(step):
        object_detection.reset_splash_watch()
        object_detection.last_bobber_height = 45  # as find_bobber() measured it
        detected_at = None
        for frame, t in list(zip(frames, times))[offset::step]:
            if object_detection.detect_splash(frame, position, mode="motion"):
                detected_at = t
                break
        assert detected_at is not None, f"missed the far bite (offset {offset})"
        assert 31.25 <= detected_at <= 31.9, f"detected at {detected_at}s (offset {offset})"


@pytest.mark.parametrize("step", [1, 2, 3])
def test_motion_splash_ignores_slowly_drifting_reflection(objects, step):
    """
    Second real capture, close bobber, NO bite: a reflection slowly drifts
    into the watch area. This used to trigger the calm-water comparison;
    with the default settings it must not count as a bite.
    """
    object_detection = objects['object_detection']
    frames, _ = load_sequence("real_drift_no_bite_sequence.npz")

    for offset in range(step):
        object_detection.reset_splash_watch()
        object_detection.last_bobber_height = 62
        results = [object_detection.detect_splash(frame, (869, 220), mode="motion")
                   for frame in frames[offset::step]]
        assert not any(results), f"false bite (offset {offset})"
