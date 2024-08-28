# Reference https://ubuntuforums.org/showthread.php?t=884556
echo "[BAFS machine specific] This script creates RAID0 on the seven Optane SSDs. Continue? [y/n]"
# Read answer until it is y or n
while true; do
    read -r answer
    if [ "$answer" = "y" ]; then
        break
    elif [ "$answer" = "n" ]; then
        exit 1
    else
        echo "Please enter y or n"
    fi
done
# sudo umount /mnt/md2
# sudo mdadm --stop /dev/md2
# sudo mdadm --zero-superblock /dev/nvme8n2 /dev/nvme17n2 /dev/nvme18n2 /dev/nvme19n2 /dev/nvme20n2 /dev/nvme21n2 /dev/nvme22n2
# sudo umount /mnt/md3
# sudo mdadm --stop /dev/md3
# sudo mdadm --zero-superblock /dev/nvme8n3 /dev/nvme17n3 /dev/nvme18n3 /dev/nvme19n3 /dev/nvme20n3 /dev/nvme21n3 /dev/nvme22n3
# cat /proc/mdstat



sudo umount /mnt/md4
sudo mdadm --stop /dev/md4
sudo mdadm --zero-superblock /dev/nvme20n2 /dev/nvme21n2 /dev/nvme22n2
sudo umount /mnt/md5
sudo mdadm --stop /dev/md5
sudo mdadm --zero-superblock /dev/nvme8n3 /dev/nvme17n3 /dev/nvme18n3 /dev/nvme19n3
cat /proc/mdstat
