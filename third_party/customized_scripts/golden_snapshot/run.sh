bash offloading.sh
mv snapshot_beginning_0.pickle offloading.pickle
python /home/kunwu2/FlashTrain/third_party/customized_scripts/memory_viz.py trace_memuse offloading.pickle >offloading.pickle.csv
bash /home/kunwu2/FlashTrain/flashtrain/tensor_cache/configs/clear_stored_tensors_bafs.sh
bash no_offloading.sh
mv snapshot_beginning_0.pickle no_offloading.pickle
python /home/kunwu2/FlashTrain/third_party/customized_scripts/memory_viz.py trace_memuse no_offloading.pickle >no_offloading.pickle.csv