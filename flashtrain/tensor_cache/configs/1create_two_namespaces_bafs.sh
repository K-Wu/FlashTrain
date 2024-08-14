echo "[BAFS machine specific] This script deletes the second namespaces and create two namespaces on each of intelmas device index 8, 9, 10, 12, 13, 14, 21 and will destroy data. Continue? [y/n]"
echo "[BAFS machine specific] The intelSSD device indices are different from those in /dev/nvme*. The /dev/nvme* numbers are 8,17,18,19,20,21,22."
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
echo "Creating namespaces on intelmas device index 8, 9, 10, 12, 13, 14, 21"
# Intelmas device index 21 has been done in the past
#sudo intelmas delete -namespace 1 -intelssd 21
#sudo intelmas create -namespace -intelssd 21 Size = 1562000000
#sudo intelmas create -namespace -intelssd 21 Size = 1562000000
#sudo intelmas attach -namespace 1 -intelssd 21
#sudo intelmas attach -namespace 2 -intelssd 21

# Do it in the reverse order so that new namespaces wont affect the indices of the smaller device indices
for i in 19 17 15 12 10 8; do #27 is already done
    sudo intelmas delete -namespace 2 -intelssd $i
    sudo intelmas create -namespace -intelssd $i Size = 750000000
    sudo intelmas create -namespace -intelssd $i Size = 750000000
    sudo intelmas attach -namespace 2 -intelssd $i
    sudo intelmas attach -namespace 3 -intelssd $i
done
