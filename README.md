 2024  AMD Pervasive AI Developer Contest - AMD  miniPC challenge
 This repository contains the source code for the 2024  AMD Pervasive AI Developer Contest, specifically for AMD  miniPC challenge.
Instructions to  execute the code:
1. Setting the RyzenAI docker using the instruction https://github.com/amd/RyzenAI-SW
2. Activate RyzenAI docker by
```
conda env list
conda activate ryzenai
```

3. DAE prediction result optimised for IPU deployment.
```
cd DAE
python newpredictDAE.py      
```

4. ResNet-50 prediction result.
```
cd ResNet-50
python newpredictResNet.py 
```
