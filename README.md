# DeepLearning Cattle

> TO-DO: Include this procediment in the requirements.txt. It was needed to use protoc later in the tutorial
```zsh
sudo apt install protobuf-compiler python-pil python-lxml python-tk
```

## Installing tensorflow object detection api
1.**Clone repository to project's root.**
```zsh
cd path_to_project && git clone https://github.com/tensorflow/models.git
```

2.**Compile with protobuf.**

```zsh
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
```

3.**Add to PYTHONPATH.**
> To make the changes permanent (not needing to type every terminal you create, DO NOT FORGET of the command source). So, if you use zsh:
```zsh
cd path_to_project/models/research && echo "export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim" >> ~/.zshrc
source ~/.zshrc
```
> OR if you use bash
```bash
cd path_to_project/models/research && echo "export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim" >> ~/.bashrc
source ~/.bashrc
```


