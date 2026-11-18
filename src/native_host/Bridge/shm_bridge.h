#ifndef SHM_BRIDGE_H
#define SHM_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Non-variadic wrapper for POSIX shm_open on Darwin/macOS ARM64.
 * Avoids variadic promotion bugs when bridging to Swift.
 */
int eyemouse_shm_open(const char *name, int oflag, int mode);

#ifdef __cplusplus
}
#endif

#endif /* SHM_BRIDGE_H */

