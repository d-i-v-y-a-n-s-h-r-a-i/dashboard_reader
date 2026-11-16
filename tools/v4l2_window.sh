#!/usr/bin/env bash
# Configure the OV9281 for the optical engine: monochrome, a small sensor
# window (not a software crop), 60 fps, manual exposure long enough to average
# the LCD backlight PWM.
#
# Run this while the recorder is NOT holding the device, or apply the same
# selection in the recorder's GStreamer pipeline before it starts.
set -euo pipefail

DEV="${1:-/dev/video0}"
W="${2:-400}"
H="${3:-200}"
FPS="${4:-60}"
EXPOSURE="${5:-40}"     # 4 ms in common UVC 100 us units; confirm on the sensor
GAIN="${6:-0}"

if ! command -v v4l2-ctl >/dev/null 2>&1; then
  echo "v4l2-ctl not found (install v4l-utils)" >&2
  exit 1
fi

echo "== formats =="
v4l2-ctl -d "$DEV" --list-formats-ext

echo "== set format =="
v4l2-ctl -d "$DEV" --set-fmt-video=width="$W",height="$H",pixelformat=GREY

echo "== set sensor crop/window =="
# The selection API is the hardware window: the sensor only sends W x H.
# Some UVC builds expose only a fixed mode; if this fails the frame-size check
# in the engine will refuse to run rather than silently rescaling.
v4l2-ctl -d "$DEV" \
  --set-selection=target=crop,flags=0,left=0,top=0,width="$W",height="$H" \
  || echo "selection API not supported; use a native sensor mode of ${W}x${H}" >&2

echo "== frame rate + manual exposure =="
v4l2-ctl -d "$DEV" --set-parm="$FPS"
v4l2-ctl -d "$DEV" --set-ctrl=auto_exposure=1 || true
v4l2-ctl -d "$DEV" --set-ctrl=exposure_absolute="$EXPOSURE" || true
v4l2-ctl -d "$DEV" --set-ctrl=gain="$GAIN" || true

echo "== verify =="
v4l2-ctl -d "$DEV" --get-selection=target=crop || true
v4l2-ctl -d "$DEV" --get-ctrl=auto_exposure,exposure_absolute,gain || true
v4l2-ctl -d "$DEV" --get-fmt-video
