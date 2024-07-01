# Get paranoid level by sysctl kernel.perf_event_paranoid
level=$(sysctl -n kernel.perf_event_paranoid)

if [ $level -eq -1 ]; then
    echo "Paranoid level is -1"
else
    echo "Paranoid level is not -1, it is $level. Setting it to -1."
    sudo sysctl -w kernel.perf_event_paranoid=-1
fi