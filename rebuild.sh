bazel --host_jvm_args=-XX:+IgnoreUnrecognizedVMOptions build -c opt --config=cuda  //tensorflow/tools/pip_package:build_pip_package --verbose_failures &&
rm -r tensorflow_pkg &&
mkdir tensorflow_pkg &&
bazel-bin/tensorflow/tools/pip_package/build_pip_package /localhd/tjin/tensorflow_latest/tensorflow/tensorflow_pkg/ &&
yes | pip uninstall tensorflow &&
pip install tensorflow_pkg/*.whl
