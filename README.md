# Cartoonizer

> Convert images and videos into a cartoon!

---

## Contents

- [Installation](#installation)
  - [Google Colab](#using-google-colab)
- [Sample Image and Video](#sample-image-and-video)

---

## Installation

### Application tested on:

- python 3.7
- tensorflow 2.1.0 
- tf_slim 1.1.0
- ffmpeg 3.4.8
- Cuda version 10.1
- OS: Linux (Ubuntu 18.04)

### Using [Google Colab](https://colab.research.google.com/drive/1oDhMEVMcsRbe7bt-2A7cDsx44KQpQwuB?usp=sharing)
1. Clone the repository using either of the below mentioned way:
   - Using Command:
        - Create a new Notebook in Colab and in the cell execute the below command.  
        
        ```
         ! git clone https://github.com/experience-ml/cartoonize.git
        ```
        
    - From Colab User Interface
 ```
        Open Colab
            └── File
                 └── Open Notebook
                          └── Github
                                └── paste the Url of the repository
 ```
 Note :  Before running the application change the runtime to GPU for processing videos but you for images CPU shall also work just fine.
 ```
            Runtime
               └── Change runtime type
                           └── Select GPU
 ```
2. After cloning the repository navigate to the `/cartoonize` using below command in the notebook cell:

   ```
   %cd cartoonize
   ```
3. Run the below commands in the notebook cell to install the requirements. 

   ```
   !pip install -r requirements.txt
   ```

4. Launch the flask app on ngrok

   ```
   !python app.py
   ```

---

## Sample Image and Video

### Emma Watson Cartoonized
<img alt="Emma Watson Cartoonized" style="border-width:0" src="static/sample_images/twitter_image.png" />

### Youtube Video of Avenger's Bar Scene Cartoonized
[![Cartoonized version of Avenger's bar scene](http://img.youtube.com/vi/GqduSLcmhto/0.jpg)](http://www.youtube.com/watch?v=GqduSLcmhto "AVENGERS BAR SCENE [Cartoonized Version]")

---

## License

1. Copyright © Cartoonizer ([Demo webapp](https://cartoonize-lkqov62dia-de.a.run.app/))

    - Authors: [Niraj Pandkar](https://twitter.com/Niraj_pandkar) and [Tejas Mahajan](https://twitter.com/tjdevWorks).

    - Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) 
    - Commercial application is prohibited by license


2. Copyright (C) Xinrui Wang, Jinze Yu. ([White box cartoonization](https://github.com/SystemErrorWang/White-box-Cartoonization))
    - All rights reserved. 
    - Licensed under the CC BY-NC-SA 4.0 
    - Also, Commercial application is prohibited license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
