# model_name=meta-llama/Llama-2-7b-hf data_path=faezeb/safety-adapt-data-v0 beaker experiment create configs/refusals/beaker-config-sft.yaml

# model_name=allenai/OLMo-7B data_path=faezeb/safety-adapt-data-v0 beaker experiment create configs/refusals/beaker-config-sft.yaml

# epochs=5 model_name=allenai/tulu-2-7b data_path=faezeb/safety-adapt-data-v0 beaker experiment create configs/refusals/beaker-config-sft.yaml

# model_name=/yizhong-tulu2-7b/ data_path=faezeb/safety-adapt-data-v0 beaker experiment create configs/refusals/beaker-config-sft.yaml

# epochs=5 model_name=/tulu-7b-uncensored/ data_path=faezeb/safety-adapt-data-v0 beaker experiment create configs/refusals/beaker-config-sft.yaml

# model_name=/yizhong-llama2-7b/ data_path=faezeb/safety-adapt-data-v0 beaker experiment create configs/refusals/beaker-config-sft.yaml


# epochs=5 model_name=allenai/tulu-2-7b beaker experiment create configs/refusals/beaker-config-sft-full-data.yaml

# epochs=5 model_name=/tulu-7b-uncensored/ beaker experiment create configs/refusals/beaker-config-sft-full-data.yaml


epochs=5 model_name=/tulu-7b-uncensored/ data_path=faezeb/tulu_mix_safety_adapt_data beaker experiment create configs/refusals/beaker-config-sft-tulu-safety-matched.yaml

epochs=5 model_name=allenai/tulu-2-7b data_path=faezeb/tulu_mix_safety_adapt_data beaker experiment create configs/refusals/beaker-config-sft-tulu-safety-matched.yaml

