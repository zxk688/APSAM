#!/bin/bash

set -e

echo "Running step 1..."
python step1_train_resnet50_SIPE.py

echo "Running step 2..."
python step2_mkcam.py

echo "Running step 3..."
python step3_mkpseudo_sam.py

echo "Running step 4..."
python step4_retrain_resunet.py

echo "All steps completed successfully."
