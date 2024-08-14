DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

source "${DIR}/_mount_all_func.sh"

# Mount the following disks if it is not already mounted.
mount_if_not /dev/md6 /mnt/md6
mount_if_not /dev/md7 /mnt/md7
