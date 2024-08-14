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
sudo mdadm --create --verbose /dev/md6 --level=stripe --raid-devices=2 /dev/nvme0n1 /dev/nvme1n1
sudo mdadm --create --verbose /dev/md7 --level=stripe --raid-devices=2 /dev/nvme7n1 /dev/nvme9n1