# Step 0. Change this to your campus ID
CAMPUSID='9084625467'
mkdir -p $CAMPUSID

python3 classifier.py \
    --use_gpu \
    --option finetune \
    --lr 1e-5\
    --seed 1234 \
    --train "data/BLM_entity_train.csv" \
    --dev "data/BLM_entity_dev.csv" \
    --test "data/BLM_entity_test.csv" \
    --dev_out "dataout/BLM_entity_dev-output.txt" \
    --test_out "dataout/BLM_entity_test-output.txt" \
    --filepath "dataout/bert-model.pt" | tee dataout/bert-train-log.txt
