## Team Information
**Team Name**: [Emergent Minds]  
**Project Title**: [Improving Customer Feedback Classification]  

## Team Members
- Name： [Xu Zhao], zID: [z5509850] 
- Name： [Jiayu Gao], zID: [z5540442]  
- Name： [Leshan Zhang], zID: [z5486819]
- Name： [Bowen Lu], zID: [z5472591]  
- Name： [Ziang Luo], zID: [z5509326]  

We have put the dataset analysis, result analysis and all representative codes into COMP6713.ipynb. You can know our work through COMP6713.ipynb, reports and presentations.

If you want to run and verify our model, you need to look at the files in the detailed folder. You need to install tranformer, pandas, torch, llamafactory libraries, and all related model parameters. Because they are too large (3g-17g), you need to go to huggingface to download them yourself. After that, you need to modify the model loading address in the relevant notebook according to the save file path of your weight model. Similarly, you also need to modify the loading path of the dataset. For the fine-tuning model, all LoRA layer weights are in the fine-tuning LoRA folder, and you can use the peft library to load the lora layer.


Note that some of the work mentioned in the individual contribution files but not in COMP6713.ipynb is located in the early contribution folder under the detailed folder.
