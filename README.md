# PyTorch DeepFloorplan

## How to run?
1. Install packages. 
```
pip install -r requirements.txt
```
2. Download r3d dataset to `dataset/r3d.tfrecords` and convert it to csv file.
```
python readTFRecord.py
python img2csv.py
```
3. Train the network,
```
python main.py
```
4. Deploy the network, 
```
python deploy.py
```