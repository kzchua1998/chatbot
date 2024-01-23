# Mini AI Chatbot
`YOLOv8` using TensorRT accelerate for faster inference for object detection, tracking and intance segmentation.

# Demo
### Vehicle Counting 
https://github.com/kzchua1998/TensorRT-Optimized-YOLOv8-for-Real-Time-Object-Tracking-and-Counting/assets/64066100/d69381b0-a4e2-48d7-a681-0eee06676639

### Human Tracking and Counting 
https://github.com/kzchua1998/TensorRT-Optimized-YOLOv8-for-Real-Time-Object-Tracking-and-Counting/assets/64066100/26feac1a-f8ea-452e-982b-b7bcb09a59f8


# Prepare the environment

1. Install python requirements.

   ``` shell
   pip install -r requirements.txt
   ```

2. Download the model [`bert-base-uncased`](https://drive.google.com/file/d/17cv-31VHBgKyqZDBzhaX-FL3xeqQCa2f/view?usp=sharing) and extract to models folder in the repository.

   ``` shell
   - chatbot
      - models
         - spec_cls.pkl
         - bert-base-uncased
   ```



# Run the Program

You can run a sanity check first to make sure all packages are installed properly before running the program

#### Sanity Check

``` shell
python predict.py
```
You can expect three job recommendations printed along with their job descriptions

#### Run Chatbot

``` shell
python main.py
```
Note: The chatbot requires a minimum of `10 words` in the description to provide recommendations.
