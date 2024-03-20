# Deploy custom trained Edge Impulse models on Himax-AI web toolkit
- This repository is a tutorial on how we can train our custom Machine learning model on Edge Impulse platform and deploy the same model on Grove Vision AI V2 using Himax AI platform.
- To run evaluations using this software, we suggest using Ubuntu 20.04 LTS environment and Google Chrome as your primary.

### The repository will follow the process:

![EI_model_Himax_AI](https://github.com/HimaxWiseEyePlus/Edge-Impulse-model-on-Himax-AI/assets/162244304/ba29d80a-104a-4ed9-bdf0-9c0adc149aa3)

## How to train model on Edge Impulse?
This section describes how you can collect data, create an impulse, generate features and finally train a model on Edge Impulse platform
### Log into the Edge Impulse Studio
Note: [You can make a Get Started here or make a free account](https://edgeimpulse.com/)

- Step 1: Create a project using Create New project button. We will name our project airpods-mario here
  
- ![Screenshot from 2024-03-20 10-17-33](https://github.com/HimaxWiseEyePlus/Edge-Impulse-model-on-Himax-AI/assets/162244304/a20854bc-74d0-496a-8aca-efa96c778c6e)

    ```
    sudo apt install make
    ```
- Step 2: Download Arm GNU Toolchain (arm-gnu-toolchain-13.2.rel1-x86_64-arm-none-eabi.tar.xz)
    ```
    cd ~
    wget https://developer.arm.com/-/media/Files/downloads/gnu/13.2.rel1/binrel/arm-gnu-toolchain-13.2.rel1-x86_64-arm-none-eabi.tar.xz
    ```
- Step 3: Extract the file

