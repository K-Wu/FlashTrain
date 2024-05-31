
# Adapted from https://serverfault.com/a/50595 and https://www.baeldung.com/linux/bash-is-directory-mounted
mount_if_not() {
    if grep -qs $2 /proc/mounts; then
        echo $2 "is mounted."
    else
        echo $2 "is not mounted."
        sudo mount $1 $2
    fi
}

# Mount /dev/md1 to /mnt/md1 if it is not already mounted.
mount_if_not /dev/md1 /mnt/md1

# The following should be mounted by default
mount_if_not /dev/md0 /mnt/raid0
mount_if_not /dev/nvme14n1 /mnt/nvme14
mount_if_not /dev/nvme7n1 /mnt/nvme7
mount_if_not /dev/nvme17n1 /mnt/nvme17