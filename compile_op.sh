cd utils/nearest_neighbors
python setup.py install --home="."
cd ../../

cd utils/cpp_wrappers
sh compile_wrappers.sh
cd ../../../

mv utils/nearest_neighbors/lib/python/KNN_NanoFLANN-0.0.0-py3.8-linux-x86_64.egg/* utils/nearest_neighbors/lib/python/
rm -r KNN_NanoFLANN-0.0.0-py3.8-linux-x86_64.egg/