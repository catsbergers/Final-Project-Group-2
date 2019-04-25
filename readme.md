# Processing xView Data - Setup Guide

### To run our code, set your "Code" folder up as follows:
1. Create a subfolder called "xView".
2. Go to https://challenge.xviewdataset.org/download-links to get the xView images and ground truth geoJSON file.
  * You must create an account first, and then log in. 
  * Logging in always happens from the “Register” page, which is not the most intuitive.
  * Once you have an account and are logged in, download the following 3 files
    * Download Training Images (zip) - results in a zip file called "train_images.zip"
    * Download Training Labels (zip) - results in a geoJSON file called "xView_train.geojson"
    * Download Validation Images (zip) - results in a zip file called "val_images.zip"
3. Move xView_train.geojson into Final-Project-Group-2/Code/xView
4. Move train_images.zip into Final-Project-Group-2/Code/xView and unzip
  * You should now have a folder called Final-Project-Group-2/Code/xView/train_images
5. Move val_images.zip into Final-Project-Group-2/Code/xView and unzip
  * You should now have a folder called Final-Project-Group-2/Code/xView/val_images
  
  
Once this is done, you should be able to run our code without having to search for and update file paths.
Note that Code/ml2_project.ipynb and Code/ml2_project_group2.py are identical, one is just a Jupyter Notebook and one is a plain Python.

### To run the Tensorflow Baseline example code, do the following:
1. Clone the xView baseline repo into the same parent folder, "Code", that you put xView in:
  * xView Baseline TensorFlow code: git clone https://github.com/DIUx-xView/baseline xview_baseline
2. Go to your brand new /xview_baseline folder
3. Create a new folder called "models" under /xview_baseline
4. Download the 3 xView pretrained models: https://github.com/DIUx-xView/baseline/releases 
  * unzip the contents into /xview_baseline/models
  * you should now see 3 folders and 3 .pb files in /xview_baseline/models, one each for vanilla, multires, and multires_aug
5. To run /xview_baseline/inference with the xView images and ground truth geoJSON you downloaded above:
  * Create a folder called "preds_output" in folder xview_baseline/
  * Then, modify file xview_baseline/inference/create_detections.sh to look like this:
  <pre>
  #!/bin/bash

TESTLIST="10.tif 100.tif 1038.tif 1040.tif"

for i in $TESTLIST; do
    echo $i
    python create_detections.py -c ../models/multires.pb -o '../preds_output/'$i'.txt' '../../xView/val_images/'$i
done
  </pre>
6. Now you can run ./create_detections.sh and see your results in xview_baseline/preds_output. 
  * Note that you can easily swap out the images you're processing and the model you are using
7. Now run xview_baseline/scoring to see how you did!!
  * Create a folder called "scores_output" in folder xview_baseline/
  * Then, modify file xview_baseline/scoring/score.bash to look like this:
  <pre>
  #!/bin/bash

set -exuo pipefail

export input_repo='../preds_output/' #$(ls ../preds_output -1 | grep -v out | grep -v labels)

export groundtruth='../../xView/' 

export userID=`echo $input_repo | cut -f 1 -d "_"`
echo "Scoring userID=$userID"

timestamp=`date +%F:%T`
mkdir -p ../scores_output/out/$timestamp

# Note ... score.py needs the trailing slash on the input path
python score.py $input_repo $groundtruth --output ../scores_output/out/$timestamp
  </pre>
8. Now you can run ./score.bash and see your results in xview_baseline/scores_output. 
  * Note that you have to run xview_baseline/inference/create_detections.sh first, otherwise there is nothing to score. 

### To run the Tensorflow Preprocessing example code, do the following:
1. Clone the xView preprocessing repo into the same parent folder, "Code", that you put xView in:
  * xView Pre-Processing code: git clone https://github.com/DIUx-xView/data_utilities xview_data_utilities
2. Go to your brand new /xview_data_utilities folder
3. The IPYNB notebook xview_data_utilities/xView Processing.ipynb is hardcoded to expect the images and ground truth to be in the same directory.  You will need to update cell 2 to look like:
<pre>
#Load an image
chip_path = '../xView/val_images/'
chip_name = '1177.tif'
arr = wv.get_image(chip_path + chip_name)
print(np.shape(arr))
plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(arr)
</pre>
4. Update cell 4 to point it to the ground truth geoJSON:
<pre>
coords, chips, classes = get_labels('../xView/xView_train.geojson')
</pre>
5. Now you can run xview_data_utilities/xView Processing.ipynb process some images!
