DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

source "${DIR}/_mount_all_func.sh"

# Mount the following disks if it is not already mounted.
mount_if_not /dev/md2 /mnt/md2
mount_if_not /dev/md3 /mnt/md3
mount_if_not /dev/md4 /mnt/md4
mount_if_not /dev/md5 /mnt/md5