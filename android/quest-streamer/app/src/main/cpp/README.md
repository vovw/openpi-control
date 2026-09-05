# Native controller source

`openpi_quest` is a NativeActivity OpenXR client. CMake requires an Android NDK,
`OPENXR_INCLUDE_DIR` (containing `openxr/openxr.h`), and an Android arm64
`OPENXR_LOADER_LIBRARY`. The APK must package the loader shared library as well
as `libopenpi_quest.so`.

The app opens a stereo GLES session and submits a dark projection layer. It
streams JSON to Android logcat tag `OpenpiXR`; diagnostics use `OpenpiXRStatus`.
The host bridge handles transport to the VR relay. No CAN or motor connection
exists in this app.

Poses use STAGE when available and LOCAL otherwise, with OpenXR right-handed
axes and xyzw quaternions. Controller grip poses receive the browser client's
default 5 cm local +Z wrist offset. The headset VIEW pose supplies yaw alignment.
Changing the tracking origin or restarting the app requires re-clutching.

Only focused sessions with fully valid, tracked headset and active, valid,
tracked controller poses emit controller data. Invalid controllers are omitted;
loss of focus emits empty controller objects while frame callbacks continue.
When Android suspends the app, the host must enforce its own timeout. The host
must never replay cached logcat frames or preserve a held clutch across a gap.

`t_monotonic_ns` and `t_unix_ns` are device timestamps, not host timestamps.
`session` changes on each launch. Button touch flags currently approximate
pressed state; they are not capacitive touch readings. There is no haptic return
channel, camera preview, boot receiver, or headset-presence override. Native
launch removes the browser session prompt, but does not guarantee tracking when
the headset is asleep, unfocused, or unable to see its controllers.
