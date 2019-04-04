# Push-Net: Deep Recurrent Neual Network for Planar Pushing Objects of Unknown Physical Properties

## What is Push-Net ?
* a deep recurrent neural network that selects actions to push objects with unknown phyical properties
* unknown physical properties: center of mass, mass distribution, friction coefficients etc.
* for technical details, refer to the [paper](http://motion.comp.nus.edu.sg/wp-content/uploads/2018/06/rss18push.pdf)

## Environment and Dependencies
* Ubuntu 14.04
* Python 2.7.6
* Pytorch 0.3.1
* GPU: GTX 980M
* CUDA version: 8.0.44
* [imutils](https://github.com/jrosebr1/imutils)


## Usage
* Input: an current input image mask of size 128 x 106, and goal specification (see [push_net_main.py](push_net_main.py))
* Output: the best push action on the image plane
* Example:
  
  input image : ```test.jpg```
  
  ```python push_net_main.py```
  
  result: the input image with the best action (red arrow) will be displayed


## Docker
Build the docker image

```docker build -t d_push_net .```

Enable client hosts to connect to X server

```xhost +```

Run nvidia docker

```nvidia-docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/<username>:/container -e DISPLAY=$DISPLAY --env QT_X11_NO_MITSHM=1 d_push_net```

If using PushNet in other python directories, must set the global PYTHONPATH variable

```export PYTHONPATH=$PYTHONPATH:/container/<path_to_PushNet>```

(Note: currently the docker file contains the path `/container/Code/PushNet`, change this line to automatically include path to PushNet on startup.)
  

## License and Citation
* See [LICENSE](LICENSE.md) file for license rights and limitations (GNU)
* If you are using part of the code for your research work, kindly cite this work

``` 
J.K. Li, D. Hsu, and W.S. Lee. Push-Net: Deep planar pushing for objects with unknown physical properties. In Proc. Robotics: Science & Systems, 2018.
```
OR bibtex: 

```
@inproceedings{Li2018PushNet,
  title={Push-Net : Deep Planar Pushing for Objects with Unknown Physical Properties},
  author={Jue Kun Li and David Hsu and Wee Sun Lee},
  booktitle={Robotics: Science and System),
  year={2018}
}
```





