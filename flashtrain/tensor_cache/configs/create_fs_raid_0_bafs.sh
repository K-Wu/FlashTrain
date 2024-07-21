# Reference https://www.digitalocean.com/community/tutorials/how-to-create-raid-arrays-with-mdadm-on-ubuntu#creating-a-raid-1-array
echo "[BAFS machine specific] This script creates EXT4 on the two RAID0 arrays on the seven Optane SSDs. Continue? [y/n]"
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

sudo mkfs.ext4 /dev/md2
sudo mkfs.ext4 /dev/md3
sudo mkdir -p /mnt/md2
sudo mkdir -p /mnt/md3