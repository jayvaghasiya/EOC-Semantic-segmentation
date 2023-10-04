# EOC-Semantic-segmentation
This repo provides the code for training the semantic segmentation model on EOC dataset.

### Steps:

  1. Installing dependencies:
     ```shell
     pip install -r requirements.txt
     ```

  2. Data preprocessing:
     ```shell
     python augment_yolo.py --source /path/to/dataset --destination /path/for/augmented/dataset
     ```

  Note: this scripts assume that you have your images and labels in single folder.

  3. After step 2 you have create datset as per this format:

          Dataset
       		   ├── train
                  ├──  images
                  ├──  labels
       		   ├── val
                  ├──  images
                  ├──  labels

  Note: images contains .jpg files and labels contains .png files.

  4. Model-training:
     ```shell
     python segformer-train.py --dataset path/to/dataset --classes /path/to/classes.csv
     ```
     Note: you can specify batch size with --batch argument default batch size is 3
     
  5. Model-Infrence;
     ```shell
     python seg-former-infrence.py --checkpoints /path/to/trained/checkpoints --source /path/to/image-or-video --classes /path/to/classes.csv
     ```






        
