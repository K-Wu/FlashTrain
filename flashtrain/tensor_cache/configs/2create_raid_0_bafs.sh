# Reference https://raid.wiki.kernel.org/index.php/RAID_setup#RAID-0
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
# sudo mdadm --create --verbose /dev/md2 --level=stripe --raid-devices=7 /dev/nvme8n2 /dev/nvme17n2 /dev/nvme18n2 /dev/nvme19n2 /dev/nvme20n2 /dev/nvme21n2 /dev/nvme22n2
# sudo mdadm --create --verbose /dev/md3 --level=stripe --raid-devices=7 /dev/nvme8n3 /dev/nvme17n3 /dev/nvme18n3 /dev/nvme19n3 /dev/nvme20n3 /dev/nvme21n3 /dev/nvme22n3
sudo mdadm --create --verbose /dev/md2 --level=stripe --raid-devices=4 /dev/nvme8n2 /dev/nvme17n2 /dev/nvme18n2 /dev/nvme19n2 
sudo mdadm --create --verbose /dev/md3 --level=stripe --raid-devices=3 /dev/nvme20n3 /dev/nvme21n3 /dev/nvme22n3
sudo mdadm --create --verbose /dev/md5 --level=stripe --raid-devices=4 /dev/nvme8n3 /dev/nvme17n3 /dev/nvme18n3 /dev/nvme19n3
sudo mdadm --create --verbose /dev/md4 --level=stripe --raid-devices=3 /dev/nvme20n2 /dev/nvme21n2 /dev/nvme22n2