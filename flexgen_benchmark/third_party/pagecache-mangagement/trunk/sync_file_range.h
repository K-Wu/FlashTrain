#define _GNU_SOURCE
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#define SYNC_FILE_RANGE_WAIT_BEFORE 1
#define SYNC_FILE_RANGE_WRITE 2
#define SYNC_FILE_RANGE_WAIT_AFTER 4

#ifdef __i386__
#define NR_sync_file_range 314
#endif

#ifdef __x86_64__
#define NR_sync_file_range 277
#endif

static inline int sync_file_range(int fd, off_t offset, off_t nbytes,
                                  int flags) {
  return syscall(NR_sync_file_range, fd, offset, nbytes, flags);
}
