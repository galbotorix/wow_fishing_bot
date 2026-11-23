from pathlib import Path
from time import time
from typing import Optional

import cv2
import mss.tools
import numpy as np
from mss import MSS
from mss.base import MSSBase

from core.screenshot_frame import ScreenshotFrame
from support.app_error_handling import InvalidMonitorError, handle_errors
from support.app_logging import get_logger
from support.constants import monitor_number_where_the_screenshot_is_taken, \
    screenshot_area, screenshots_output_dir_name, capture_full_screen, auto_select_monitor_from_game_window
from support.window_utils import get_game_window, get_window_rect


logger = get_logger(__name__)


class ScreenshotCapture:
    def __init__(self):
        # Choose where screenshots are saved; created automatically if it doesn't exist
        project_root = Path(__file__).parent.parent
        self.output_dir = project_root / screenshots_output_dir_name
        self.output_dir.mkdir(exist_ok=True)
        self._last_logged_monitor: Optional[int] = None

    @staticmethod
    def monitor_index_for_point(monitors: list[dict], x: int, y: int) -> Optional[int]:
        """
        Index (1..n, mss numbering) of the monitor containing screen point
        (x, y), or None. monitors[0] is mss's "all monitors" box and is skipped.
        """
        for index, mon in enumerate(monitors[1:], start=1):
            if mon["left"] <= x < mon["left"] + mon["width"] and mon["top"] <= y < mon["top"] + mon["height"]:
                return index
        return None

    def __choose_monitor_number(self, sct: MSSBase) -> int:
        """
        The monitor the game window is on (by the window's centre), when
        auto_select_monitor_from_game_window is on and a game window is
        known; otherwise monitor_number_where_the_screenshot_is_taken.
        """
        chosen = monitor_number_where_the_screenshot_is_taken
        reason = "configured"
        if auto_select_monitor_from_game_window:
            rect = get_window_rect(get_game_window())
            if rect is not None:
                left, top, right, bottom = rect
                index = self.monitor_index_for_point(sct.monitors, (left + right) // 2, (top + bottom) // 2)
                if index is not None:
                    chosen, reason = index, "game window is on it"
        if chosen != self._last_logged_monitor:
            mon = sct.monitors[chosen] if 0 <= chosen < len(sct.monitors) else {}
            logger.info("Capturing monitor %d (%s): %s", chosen, reason,
                        {k: mon.get(k) for k in ("left", "top", "width", "height")})
            self._last_logged_monitor = chosen
        return chosen

    @staticmethod
    def __get_information_of_monitor(
        sct: MSSBase, monitor_number: int = monitor_number_where_the_screenshot_is_taken
    ) -> dict:
        """
        Returns mss's monitor info dict (top/left/width/height/...) for
        the given monitor index, reusing an already-open MSS session.
        """
        try:
            mon = sct.monitors[monitor_number]
        except IndexError as error:
            raise InvalidMonitorError(
                f"Invalid monitor_number {monitor_number}; "
                f"available monitors: 0..{len(sct.monitors) - 1}"
            ) from error

        logger.debug("Monitor %s info: %s", monitor_number, mon)
        return mon

    @handle_errors()
    def capture_screenshot(self, screenshot_number: int, save: bool = True) -> ScreenshotFrame:
        """
        Grabs the configured monitor/region. With save=False the image is
        only kept in memory (much faster -- writing a 1080p PNG is most of
        the cost of a capture), which lets the splash watch sample often.
        """
        with MSS() as sct:
            # Get information of the configured monitor (single session,
            # reused for every grab below).
            monitor_number = self.__choose_monitor_number(sct)
            mon = self.__get_information_of_monitor(sct, monitor_number)

            # Capture either the entire monitor, or a cropped region
            # offset from the monitor's top-left corner by the
            # configured pixel amounts, depending on capture_full_screen.
            if capture_full_screen:
                region = {
                    "top": mon["top"],
                    "left": mon["left"],
                    "width": mon["width"],
                    "height": mon["height"],
                }
            else:
                region = {
                    "top": mon["top"] + screenshot_area["pixels_from_top"],
                    "left": mon["left"] + screenshot_area["pixels_from_left"],
                    "width": screenshot_area["width"],
                    "height": screenshot_area["height"],
                }

            # Include the iteration index so each screenshot gets a unique filename
            filename = "sct-mon{mon}_{top}x{left}_{width}x{height}_{screenshot_number}.png".format(
                mon=monitor_number, screenshot_number=screenshot_number, **region
            )
            output = self.output_dir / filename

            # Grab the data
            sct_img = sct.grab(region)

            # Save to the picture file
            if save:
                mss.tools.to_png(sct_img.rgb, sct_img.size, output=str(output))
                logger.info("Saved screenshot: %s", output)

            # yield the single screenshot
            return ScreenshotFrame(
                image=cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR),
                timestamp=time(),
                origin=(region["left"], region["top"])
            )

    @handle_errors()
    def clear_screenshots(self) -> None:
        """
        Deletes every screenshot saved so far in output_dir. Intended to
        be called when the bot stops, since the screenshots are only a
        debugging/analysis aid during a run and can pile up quickly
        (one file every wait_time_between_screenshots while watching
        for a splash).
        """
        deleted_count = 0
        for png_file in self.output_dir.glob("*.png"):
            png_file.unlink()
            deleted_count += 1
        logger.info("Cleared %d screenshot(s) from %s", deleted_count, self.output_dir)
