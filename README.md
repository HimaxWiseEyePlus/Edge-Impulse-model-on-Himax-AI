# Deploy custom trained Edge Impulse models on Himax-AI web toolkit
- This repository is a tutorial on how we can train our custom Machine learning model on Edge Impulse platform and deploy the same model on Grove Vision AI V2 using Himax AI platform.
- To run evaluations using this software, we suggest using Ubuntu 20.04 LTS environment and Google Chrome as your primary.

### The repository will follow the process:

![EI_model_Himax_AI](https://github.com/HimaxWiseEyePlus/Edge-Impulse-model-on-Himax-AI/assets/162244304/ba29d80a-104a-4ed9-bdf0-9c0adc149aa3)

## How to train model on Edge Impulse?
This section describes how you can collect data, create an impulse, generate features and finally train a model on Edge Impulse platform. Our public project can be found [here](https://studio.edgeimpulse.com/public/372849/live).
### Log into the Edge Impulse Studio
Note: [You can make a Get Started here or make a free account](https://edgeimpulse.com/)

- Step 1: Create a project using Create New project button. We will name our project airpods-mario here

![Screenshot from 2024-04-01 10-49-45](https://github.com/HimaxWiseEyePlus/Edge-Impulse-model-on-Himax-AI/assets/162244304/b8fd3daa-cd84-430b-9cef-871aa1eee23e)

- Step 2: Project goal: For the purpose of this tutorial, we'll be building an object detection use case to detection two objects: a cup and a computer mouse.

![media_20240401_105802_1302071064980971357 (1)](https://github.com/HimaxWiseEyePlus/Edge-Impulse-model-on-Himax-AI/assets/162244304/1635a5ba-8700-4e0c-b6c5-b603abafc31f)![media_20240401_105802_5329619283457749483 (1)](https://github.com/HimaxWiseEyePlus/Edge-Impulse-model-on-Himax-AI/assets/162244304/d6d7a67d-495f-484a-9557-753c6546d89f)

- Step 3: Data collection

Here we will be utilizing Edge Impulse's data acquisition feature to collect data for our two classes. Alternatively, custom datasets can also be uploaded onto the platform using the 'Add existing data' option or an organizational data bucket can be connected as well.

![Screenshot from 2024-04-01 11-10-51](https://github.com/HimaxWiseEyePlus/Edge-Impulse-model-on-Himax-AI/assets/162244304/1099e569-2a89-49d8-b44a-cb71aec57952)

In our case, 219 photos of both classes were captured and labelled collectively. A ratio of 82%/18% was used for training and test respectively.

- Step 4: Train the object detection model

  - Designing an impulse:
    An impulse is a pipeline used to define the use-case flow. It takes in the images, performs feature engineering and uses a learning block to perform the desired task. A comprehensive understanding and other applications can be found [here](https://www.youtube.com/watch?v=o8UG1TJXuwk)

    Starting off with our from the dataset collected. We will resize them to 160x160 pixels. This will be the input to our Transfer Learning block.

    ![Screenshot from 2024-04-01 15-40-34](https://github.com/HimaxWiseEyePlus/Edge-Impulse-model-on-Himax-AI/assets/162244304/4313104a-06c2-4269-a8a2-f3247bf4c243)

  - Generating features
    The feature explorer stage helps a developer to understand and analyze their dataset. The graph on the right displays a dimensionally reduced form of our input data(images). Not only this is helpful to feed features into the model that will help with the training but also helps one understand the relationships between different classes.

    ![Screenshot from 2024-04-01 15-49-23](https://github.com/HimaxWiseEyePlus/Edge-Impulse-model-on-Himax-AI/assets/162244304/e0092b2d-dd74-4f28-8198-c67e6c031f96)

  - Training model
    Edge Impulse studio has the support to train multiple model architectures such as MobileNetV1, YoloV5 or YoloX. Alternatively, a user can even 'bring their own model'. For this tutorial, we trained a YoloV5 based on Ultralytics YOLOv5 which supports RGB input at any resolution(square images only).

    Hyperparameters are as follows:

    - Number of epochs: 20
    - Model size: Nano
    - Batch size: 32
   
    Note: It is important to note that there's a 20 min time limit for training for the community version
 
    ![Screenshot from 2024-04-01 16-28-43](https://github.com/HimaxWiseEyePlus/Edge-Impulse-model-on-Himax-AI/assets/162244304/39220d6e-add5-4225-b6d1-a6c7552c3ee9)

    Once the model is trained, it can be downloaded from the dashboard. In order to be accelerated by the Ethos-U NPU the network operators must be quantised to either 8-bit (unsigned or signed) or 16-bit (signed).
    
    ![Screenshot from 2024-04-01 16-28-43](https://github.com/HimaxWiseEyePlus/Edge-Impulse-model-on-Himax-AI/assets/162244304/1bc5d71a-e06f-4877-b758-bbcb234b809d)

## Model conversion to vela

  [Vela](https://pypi.org/project/ethos-u-vela/), is a tool used to compile a TensorFlow Lite for Microcontrollers neural network model into an optimised version that can run on an embedded system containing an Arm Ethos-U NPU. In order to run flash the model onto the Grove Vision AI v2, we need to convert the tflite int8 file downloaded from Edge Impulse to a _vela.tflite file.

  We'll do this on [Google CoLab](https://colab.research.google.com/). Once you have the notebook ready, upload your int8 quantised model to Google CoLab using the Upload button on the sidebar.
  
  <p align="center">
    <img src=![Screenshot from 2024-04-01 16-20-27](https://github.com/HimaxWiseEyePlus/Edge-Impulse-model-on-Himax-AI/assets/162244304/f7f7fb78-925d-4be1-ba3d-223ea2d1c593) />
  </p>
  
  In code cells, run the following lines:
  
  ```
  !pip install ethos-u-vela
  ```
   And then:
  ```
  !vela [your_model_name].tflite --accelerator-config ethos-u55-64
  ```
  Downloaded the converted tflite model from under the output folder.

  ![Screenshot from 2024-04-01 16-20-27](https://github.com/HimaxWiseEyePlus/Edge-Impulse-model-on-Himax-AI/assets/162244304/5f2eaa01-2514-47b4-a55e-2e7ec23aa6c7)

## scenario_app post processing

Build firmware and flash the image for our object detection model onto the Grove Vision AI v2 will be heavily inspired and referenced from one of our other repositories, [Seeed_Grove_Vision_AI_Module_V2](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2).

To build a new use-case from now on referred to as scenario app like YOLOv5 object detection, one can build their own application in the [scenario_app](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2/tree/main/EPII_CM55M_APP_S/app/scenario_app). To run the build the firmware for the YOLOv5 model, we have made an `APP_TYPE` called `ei_yolov5_od`.

- To run this scenario_app, change the `APP_TYPE` to `ei_yolov5_od`.
  ```
  APP_TYPE = ei_yolov5_od
  ```
- Build the firmware reference the part of [Build the firmware at Linux environment](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2?tab=readme-ov-file#build-the-firmware-at-linux-environment)
- Connect Grove Vision AI v2 to your computer.
- Make sure `Minicom` is disconnected.
- Grant permission to access the device:
  ```
  sudo setfacl -m u:[USERNAME]:rw [COM NUMBER]
  ```
  Where the [USERNAME] is the username of the computer and the [COM NUMBER] is the COM number of your `SEEED Grove Vision AI v2`. An example is `sudo setfacl -m u:kris:rw /dev/ttyACM0`

  Note: Use Google Chrome browser

- Open `Terminal` and key-in following command
    - port: the COM number of your `Seeed Grove Vision AI Module V2`, for example,`/dev/ttyACM0`
    - baudrate: 921600
    - file: your firmware image [maximum size is 1MB]
    - model: you can burn multiple models "[model tflite] [position of model on flash] [offset]"
      - Position of model on flash is defined at [~/tflm_fd_fm/common_config.h](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2/blob/main/EPII_CM55M_APP_S/app/scenario_app/tflm_fd_fm/common_config.h#L18)
    ```
    python3 xmodem/xmodem_send.py --port=[your COM number] --baudrate=921600 --protocol=xmodem --file=we2_image_gen_local/output_case1_sec_wlcsp/output.img --model="model_zoo/tflm_fd_fm/0_fd_0x200000.tflite 0x200000 0x00000" --model="model_zoo/tflm_fd_fm/1_fm_0x280000.tflite 0x280000 0x00000"  --model="model_zoo/tflm_fd_fm/2_il_0x32A000.tflite 0x32A000 0x00000"
    ```
    - It will start to burn firmware image and model automatically.
  -  Press `reset` buttun on `Seeed Grove Vision AI Module V2` and it will success to run the algorithm.
  

Follow the remaining steps from 
## Running on Himax AI


