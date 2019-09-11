# Unet
Implementation of Unet model using Keras libraries to segment the fibers from artificial SEM images

This project is divided into two part:

1. Create the artifical SEM images and their corresponding annotation

      i) Go to folder image_synthesis
      
      ii) Run the "image_systhesis.py" - # This program need to pass some command line argument
      
      iii) Follow the cmd line instruction as show in "run_image_systhesis_frm_cmd" file
      
      iv) This will create you "images" and "masks" for train
      
      v) Run the code again to generated images for test purpose
  
2. Implementation of Unet to segment the fibers
      i) Copy the "train" and "test" folder to the running directory

        Following is the directory structure
              Unet
              |--Train
                  |--images
                  |--masks
              |--Test
                  |--images
              |-- Outputs
              |--demo.ipynb
              |--main.py
              |--model.py

      ii) Run the demo.ipynb or main.py file to see the implementation
      
The model.py file contains the Unet model
