# EVAL
The VQA eval pipeline is called from eval_vqa.py and consists of the following steps:
- The data is loaded using utils/load_vid.py and data/dataloader.py 
- After data loading the model wrapper class (data/model_loader.py) wraps a VLM 
- The wrapper also prepares the inputs and is called for the forward pass.
- The decoded outputs are stored in a csv file

The next step in the pipeline is to evaluate the produced answer using a judge. We use batch API to obtain faster evaluations. 


