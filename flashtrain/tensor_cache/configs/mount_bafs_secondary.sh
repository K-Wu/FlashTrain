DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

source "${DIR}/_mount_all_func.sh"

# Mount /dev/md1 to /mnt/md1 if it is not already mounted.
# mount_if_not /dev/md1 /mnt/md1

# The following should be mounted by default
mount_if_not /dev/md0 /mnt/raid0
mount_if_not /dev/nvme14n1 /mnt/nvme14
mount_if_not /dev/nvme7n1 /mnt/nvme7
mount_if_not /dev/nvme17n1 /mnt/nvme17