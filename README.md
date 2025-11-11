# MomeDTA

This is the official repository of "MomeDTA: Improving Generalizability in Drug-Target Affinity Prediction by Mixture of Multi-view Experts" accepted by NeurIPS 2025 Workshop: MLSB.

## Requirements

You can set up the environment for running MomeDTA by
```
conda env create -f environment.yml
```

## Usage

### Preprocess

Please refer to `preprocess/README.md`.

### Training
You can train the model by 
```
python script/momedta.py
```
After the program finishes, you can see the outputs in `output/`.

If you want to resume from a previously trained model, please refer to `script/resume.sh`.

### Test

Please refer to `script/test.sh` if you want to exclusively test a model.

## References

Our code is based on [coach-pl](https://github.com/DuskNgai/coach-pl). We thank the authors for their great foundational work.

## Citation

TBD
