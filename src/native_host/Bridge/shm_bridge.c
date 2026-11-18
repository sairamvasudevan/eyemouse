#include "shm_bridge.h"
#include <sys/mman.h>
#include <fcntl.h>

int eyemouse_shm_open(const char *name, int oflag, int mode) {
    return shm_open(name, oflag, (mode_t)mode);
}

