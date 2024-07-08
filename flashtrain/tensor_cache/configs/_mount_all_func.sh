
# Adapted from https://serverfault.com/a/50595 and https://www.baeldung.com/linux/bash-is-directory-mounted
mount_if_not() {
    if grep -qs $2 /proc/mounts; then
        echo $2 "is mounted."
    else
        echo $2 "is not mounted."
        sudo mount -o data=ordered $1 $2
    fi
}